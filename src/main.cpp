#include <Geode/Geode.hpp>
#include <Geode/modify/PlayLayer.hpp>
#include <windows.h>
#include <string>
#include <thread>
#include <atomic>
#include <set>
#include <map>

using namespace geode::prelude;

const float LEARNING_RATE = 0.01f;

struct GameData{
    float playerY;
    float velocityY;
    float nearestObstacleX;
    float nearestObstacleY;
    float obstacleHeight;
    float obstacleWidth;
    bool isGrounded;
    bool isShip;
    float reward = 0.0f;
};


class $modify(DashAIPlayLayer, PlayLayer) {
    struct Fields {
        cocos2d::CCLabelBMFont* label = nullptr;
        GameData *gameData = nullptr; // Will be initialized in init
        float currentDistance = 0.0f;
        float bestDistance = 0.0f;
        int currentRunLength = 0; // in frames
        std::vector<int> runLengths;
        std::set<float> passedObstaclePositions; // Track X positions of passed obstacles
        std::map<float, bool> pendingRewards; // Obstacle X -> isNew (for deferred rewards)
        bool lastFrameWasGrounded = false;
        bool jumpedThisFrame = false;
        int framesInAir = 0; // Track how long we've been airborne
        bool wasUnnecessaryJump = false; // Flag unnecessary jumps
        float lastPlayerX = 0.0f; // For distance-based rewards
        bool deathPenaltyApplied = false; // Track if death penalty was already applied
    };
    
    bool init(GJGameLevel* level, bool useReplay, bool dontCreateObjects) {
        if (!PlayLayer::init(level, useReplay, dontCreateObjects)) {
            return false;
        }
        
        // Create and add the label showing best run stats
        auto* label = cocos2d::CCLabelBMFont::create("Best: 0", "bigFont.fnt");
        label->setScale(0.8f);
        
        auto winSize = cocos2d::CCDirector::sharedDirector()->getWinSize();
        label->setPosition({winSize.width / 2, winSize.height - 30.f});
        
        m_fields->currentDistance = 0.0f;
        m_fields->bestDistance = 0.0f;
        m_fields->currentRunLength = 0.0f;
        
        this->addChild(label, 1000);
        m_fields->label = label;

        // Initialize the neural network
        if (!m_fields->gameData) {
            m_fields->gameData = new GameData();
        }
        
        return true;
    }

    GameObject* getNearestObstacle(PlayerObject* player) {
        float minDist = 1e9f;
        auto playerPos = player->getPosition();
        // m_hazardCollisionObjects is inherited from GJBaseGameLayer
        GameObject* nearest = nullptr;
        for (auto obj : this->m_hazardCollisionObjects) {
            if (!obj) continue;
            auto objPos = obj->getPosition();
            float dx = objPos.x - playerPos.x;
            float dy = objPos.y - playerPos.y;
            float dist = sqrtf(dx * dx + dy * dy);
            if (dist < minDist) {
                minDist = dist;
                nearest = obj;
            }
        }
        return nearest; // -1 if no hazard found
    }

    class PipeClient {
        HANDLE handle = INVALID_HANDLE_VALUE;
    public:
        bool connect() {
            if (handle != INVALID_HANDLE_VALUE) return true;
            // Try to connect with non-blocking behavior
            handle = CreateFileA(R"(\\.\pipe\dashai)", GENERIC_READ | GENERIC_WRITE, 0, nullptr, OPEN_EXISTING, FILE_FLAG_OVERLAPPED, nullptr);
            if (handle == INVALID_HANDLE_VALUE) return false;
            
            // Set pipe mode to non-blocking for reads
            DWORD mode = PIPE_READMODE_BYTE | PIPE_NOWAIT;
            if (!SetNamedPipeHandleState(handle, &mode, nullptr, nullptr)) {
                CloseHandle(handle);
                handle = INVALID_HANDLE_VALUE;
                return false;
            }
            return true;
        }
        bool isConnected() const { return handle != INVALID_HANDLE_VALUE; }
        void close() {
            if (handle != INVALID_HANDLE_VALUE) {
                CloseHandle(handle);
                handle = INVALID_HANDLE_VALUE;
            }
        }
        bool send(const std::string& line) {
            if (!isConnected()) return false;
            DWORD written = 0;
            std::string payload = line + "\n";
            BOOL ok = WriteFile(handle, payload.data(), (DWORD)payload.size(), &written, nullptr);
            if (!ok) {
                close();
                return false;
            }
            return written == payload.size();
        }
        bool recvLine(std::string& out) {
            if (!isConnected()) return false;
            char buf[16] = {0};
            DWORD read = 0;
            BOOL ok = ReadFile(handle, buf, sizeof(buf)-1, &read, nullptr);
            if (!ok || read == 0) {
                if (GetLastError() == ERROR_NO_DATA) {
                    return false; // No data available (non-blocking)
                }
                close();
                return false;
            }
            buf[read] = 0;
            out = std::string(buf);
            // Remove trailing newlines
            out.erase(std::remove(out.begin(), out.end(), '\n'), out.end());
            out.erase(std::remove(out.begin(), out.end(), '\r'), out.end());
            return true;
        }
        ~PipeClient() { close(); }
    };

    void resetLevel() {
        // Clear pending rewards when level resets (death/restart)
        m_fields->pendingRewards.clear();
        m_fields->jumpedThisFrame = false;
        m_fields->lastFrameWasGrounded = false;
        m_fields->deathPenaltyApplied = false;
        PlayLayer::resetLevel();
    }

    void onQuit() {
        // Save run length when the run ends (death/quit)
        if (m_fields->currentRunLength > 0) {
            m_fields->runLengths.push_back(m_fields->currentRunLength);
            // Keep only last 100 runs
            if (m_fields->runLengths.size() > 100) {
                m_fields->runLengths.erase(m_fields->runLengths.begin());
            }
            m_fields->currentRunLength = 0;
        }
        PlayLayer::onQuit();
    }

    void  postUpdate(float dt) {
        PlayLayer::postUpdate(dt);
        static float lastObstacleX = 0.0f;
        static bool firstFrame = true;
        static PipeClient pipe;
        static std::atomic<int> lastAction{0};
        static bool threadStarted = false;
        // Hold timing: 0=short(0.1s), 1=medium(0.2s), 2=long(0.35s), 3=extra-long(0.5s)
        static float holdDuration = 0.0f;
        static float holdTimer = 0.0f;
        static bool isHolding = false;
        auto nearestObj = getNearestObstacle(this->m_player1);
        float reward = 0.0f; // no reward for being alive
        if (!threadStarted) {
            threadStarted = true;
            PipeClient* pipePtr = &pipe;
            std::atomic<int>* actionPtr = &lastAction;
            std::thread([pipePtr, actionPtr]() {
                while (true) {
                    if (!pipePtr->isConnected()) pipePtr->connect();
                    std::string line;
                    if (pipePtr->recvLine(line)) {
                        int action = std::atoi(line.c_str());
                        *actionPtr = action;
                    } else {
                        std::this_thread::sleep_for(std::chrono::milliseconds(1));
                    }
                }
            }).detach();
        }
        if (nearestObj != nullptr) {
            auto playerPos = this->m_player1->getPosition();
            auto objPos = nearestObj->getPosition();
            m_fields->gameData->playerY = playerPos.y;
            m_fields->gameData->velocityY = this->m_player1->m_yVelocity;
            m_fields->gameData->nearestObstacleX = objPos.x - playerPos.x;
            m_fields->gameData->nearestObstacleY = objPos.y - playerPos.y;
            m_fields->gameData->obstacleHeight = nearestObj->m_height;
            m_fields->gameData->obstacleWidth = nearestObj->m_width;
            m_fields->gameData->isGrounded = this->m_player1->m_isOnGround;
            m_fields->gameData->isShip = this->m_player1->m_isShip;
            
            // Detect jump in cube mode (transitioned from grounded to not grounded)
            if (!this->m_player1->m_isShip && m_fields->lastFrameWasGrounded && !this->m_player1->m_isOnGround) {
                m_fields->jumpedThisFrame = true;
                m_fields->framesInAir = 0;
                m_fields->wasUnnecessaryJump = true; // Assume unnecessary until proven otherwise
            }
            
            // Track frames in air
            if (!this->m_player1->m_isOnGround && !this->m_player1->m_isShip) {
                m_fields->framesInAir++;
            } else if (this->m_player1->m_isOnGround) {
                m_fields->framesInAir = 0;
                m_fields->wasUnnecessaryJump = false; // Reset on landing
            }
            
            // Better reward structure - defer reward until safe distance cleared
            if (!firstFrame && lastObstacleX > 0 && m_fields->gameData->nearestObstacleX <= 0 && !this->m_player1->m_isDead) {
                // Obstacle was just passed - add to pending rewards
                float obstacleX = objPos.x;
                
                // Check if this obstacle position has been passed before
                bool isNew = (m_fields->passedObstaclePositions.find(obstacleX) == m_fields->passedObstaclePositions.end());
                m_fields->pendingRewards[obstacleX] = isNew;
                m_fields->passedObstaclePositions.insert(obstacleX);
                
                // Jump was necessary - clear the flag
                m_fields->wasUnnecessaryJump = false;
            }
            
            // Immediate reward for forward progress (helps learn even before first obstacle)
            float currentX = playerPos.x;
            if (currentX > m_fields->lastPlayerX) {
                reward += (currentX - m_fields->lastPlayerX) * 1.0f;  // Reward per unit moved forward
            }
            m_fields->lastPlayerX = currentX;
            
            m_fields->lastFrameWasGrounded = this->m_player1->m_isOnGround;
            // Small reward for forward progress
            //reward += 0.1f * dt;
            
            lastObstacleX = m_fields->gameData->nearestObstacleX;
            firstFrame = false;
        }
        
        // Process pending rewards - only grant after clearing safe distance
        auto playerPos = this->m_player1->getPosition();
        std::vector<float> rewardsToRemove;
        for (auto& pair : m_fields->pendingRewards) {
            float obstacleX = pair.first;
            bool isNew = pair.second;
            
            // Check if player has cleared 10 units past the obstacle
            if (playerPos.x - obstacleX > 10.0f) {
                if (isNew) {
                    reward += 1000.0f;  // Large reward for passing new obstacle
                    geode::log::info("Rewarded for passing new obstacle (safe)");
                } else {
                    reward += 200.0f;  // Smaller reward for re-passing
                }
                rewardsToRemove.push_back(obstacleX);
            }
        }
        // Remove processed rewards
        for (float obstacleX : rewardsToRemove) {
            m_fields->pendingRewards.erase(obstacleX);
        }
        
        // Add death penalty only once when player first dies
        if (this->m_player1->m_isDead && !m_fields->deathPenaltyApplied) {
            reward -= 100.0f;  // Penalty for death (applied once)
            m_fields->deathPenaltyApplied = true;
        } else if (!this->m_player1->m_isDead) {
            m_fields->deathPenaltyApplied = false;  // Reset when alive again
        }
        
        m_fields->gameData->reward = reward;
        
        // Track distance (use player X position as distance traveled)
        playerPos = this->m_player1->getPosition();
        m_fields->currentDistance = playerPos.x;
        m_fields->currentRunLength++; // increment frame counter
        
        // Update best distance
        if (m_fields->currentDistance > m_fields->bestDistance) {
            m_fields->bestDistance = m_fields->currentDistance;
            
            // Calculate average run length in frames
            float avgRunLength = 0.0f;
            if (!m_fields->runLengths.empty()) {
                int sum = 0;
                for (int length : m_fields->runLengths) {
                    sum += length;
                }
                avgRunLength = (float)sum / m_fields->runLengths.size();
            }
            
            // Get level length for percentage calculation
            float levelLength = this->m_levelLength;
            float bestPercent = (m_fields->bestDistance / levelLength) * 100.0f;
            float avgPercent = (avgRunLength / (m_fields->bestDistance > 0 ? m_fields->bestDistance : 1.0f)) * 100.0f;
            
            // Update label with best distance and average run length as percentages
            char labelText[128];
            snprintf(labelText, sizeof(labelText), "Best: %.1f%% | Avg: %.1f%%", bestPercent, avgPercent);
            m_fields->label->setString(labelText);
        }
        
        char buffer[256];
        snprintf(buffer, sizeof(buffer), "%f,%f,%f,%f,%f,%f,%d,%d,%f,%d",
            m_fields->gameData->playerY,
            m_fields->gameData->velocityY,
            m_fields->gameData->nearestObstacleX,
            m_fields->gameData->nearestObstacleY,
            m_fields->gameData->obstacleHeight,
            m_fields->gameData->obstacleWidth,
            m_fields->gameData->isGrounded ? 1 : 0,
            m_fields->gameData->isShip ? 1 : 0,
            m_fields->gameData->reward,
            this->m_player1->m_isDead ? 1 : 0);
        if (!pipe.isConnected()) pipe.connect();
        pipe.send(buffer);
        
        // Simplified binary action: 0=no jump, 1=jump (medium hold)
        int action = lastAction.load();
        const float mediumHoldDuration = 0.2f;
        
        // Action 1 = jump
        if (!isHolding && action == 1) {
            isHolding = true;
            holdDuration = mediumHoldDuration;
            holdTimer = 0.0f;
            this->m_player1->pushButton(PlayerButton::Jump);
            lastAction = 0; // Reset to no-jump
        }
        
        // Update hold timer
        if (isHolding) {
            holdTimer += dt;
            if (holdTimer >= holdDuration) {
                // Release the button after hold duration
                this->m_player1->releaseButton(PlayerButton::Jump);
                isHolding = false;
                holdTimer = 0.0f;
            }
        }
    }
};
