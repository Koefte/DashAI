#include <Geode/Geode.hpp>
#include <Geode/Prelude.hpp>
#include <Geode/modify/PlayerObject.hpp>
#include <Geode/modify/PlayLayer.hpp>
#include <Geode/binding/PlayLayer.hpp>
#include <Geode/binding/GameObject.hpp>
#include <fstream>
#include <chrono>
#include <sstream>
#include <algorithm>
#include <windows.h>

using namespace geode::prelude;

void writeLog(const std::string& message) {
    std::ofstream file("C:\\Users\\henkv\\Documents\\CodeProjects\\DashAI\\src\\debug_output.txt", std::ios::app);
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    file << "[" << std::ctime(&time) << "] " << message << "\n";
    file.close();
}

struct Observation {
    double timeSeconds;
    cocos2d::CCPoint position;
    double yVelocity;
    float playerSpeed;
    float levelPercent;
    float distToEnd;
    float distToNearest;
    bool isUpsideDown;
    bool isOnGround;
    bool isDashing;
    bool isShip;
    bool isBall;
    bool isDart;
    bool isRobot;
    bool isSpider;
    bool isSwing;
};

float nearestObstacleDistance(PlayLayer* pl, const cocos2d::CCPoint& playerPos) {
    if (!pl) return 9999.f;
    float best = 9999.f;
    auto considerArray = [&](cocos2d::CCArray* arr) {
        if (!arr) return;
        CCObject* objRaw = nullptr;
        CCARRAY_FOREACH(arr, objRaw) {
            auto* obj = static_cast<GameObject*>(objRaw);
            if (!obj) continue;
            const float dx = obj->getPositionX() - playerPos.x;
            if (dx > 0.f && dx < best) best = dx;
        }
    };
    auto considerVec = [&](const gd::vector<GameObject*>& vec) {
        for (auto* obj : vec) {
            if (!obj) continue;
            const float dx = obj->getPositionX() - playerPos.x;
            if (dx > 0.f && dx < best) best = dx;
        }
    };

    considerVec(pl->m_activeSaveObjects1);
    considerVec(pl->m_activeSaveObjects2);
    considerVec(pl->m_dynamicSaveObjects);
    considerArray(pl->m_checkpointArray);
    return best;
}

Observation buildObservation(PlayerObject* player, double timeSeconds) {
    Observation obs{};
    obs.timeSeconds = timeSeconds;
    obs.position = player->m_position;
    obs.yVelocity = player->m_yVelocity;
    obs.playerSpeed = player->m_playerSpeed;
    if (auto pl = PlayLayer::get()) {
        obs.levelPercent = pl->getCurrentPercent();
        obs.distToEnd = pl->m_endXPosition - player->m_position.x;
        obs.distToNearest = nearestObstacleDistance(pl, player->m_position);
    } else {
        obs.levelPercent = 0.f;
        obs.distToEnd = 0.f;
        obs.distToNearest = 9999.f;
    }
    obs.isUpsideDown = player->m_isUpsideDown;
    obs.isOnGround = player->m_isOnGround;
    obs.isDashing = player->m_isDashing;
    obs.isShip = player->m_isShip;
    obs.isBall = player->m_isBall;
    obs.isDart = player->m_isDart;
    obs.isRobot = player->m_isRobot;
    obs.isSpider = player->m_isSpider;
    obs.isSwing = player->m_isSwing;
    return obs;
}

std::string observationToCSV(const Observation& obs) {
    std::ostringstream ss;
    ss << obs.timeSeconds << ','
       << obs.position.x << ','
       << obs.position.y << ','
       << obs.yVelocity << ','
       << obs.playerSpeed << ','
         << obs.levelPercent << ','
         << obs.distToEnd << ','
         << obs.distToNearest << ','
       << obs.isUpsideDown << ','
       << obs.isOnGround << ','
       << obs.isDashing << ','
       << obs.isShip << ','
       << obs.isBall << ','
       << obs.isDart << ','
       << obs.isRobot << ','
       << obs.isSpider << ','
       << obs.isSwing;
    return ss.str();
}

struct PipeClient {
    HANDLE handle = INVALID_HANDLE_VALUE;
    bool connect() {
        if (handle != INVALID_HANDLE_VALUE) return true;
        handle = CreateFileA(R"(\\.\pipe\dashai)", GENERIC_WRITE | GENERIC_READ, 0, nullptr, OPEN_EXISTING, FILE_FLAG_OVERLAPPED, nullptr);
        return handle != INVALID_HANDLE_VALUE;
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
        const std::string payload = line + "\n";
        const BOOL ok = WriteFile(handle, payload.data(), static_cast<DWORD>(payload.size()), &written, nullptr);
        if (!ok) {
            close();
            return false;
        }
        return written == payload.size();
    }
    bool tryReadLine(std::string& out) {
        if (!isConnected()) return false;
        DWORD available = 0;
        if (!PeekNamedPipe(handle, nullptr, 0, nullptr, &available, nullptr)) {
            close();
            return false;
        }
        if (available == 0) return false;
        std::string buf;
        buf.resize(available);
        DWORD read = 0;
        if (!ReadFile(handle, buf.data(), available, &read, nullptr)) {
            close();
            return false;
        }
        buf.resize(read);
        out = buf;
        return true;
    }
};

class $modify(PlayerObject) {
    struct Fields {
        int m_totalJumps = 0;
        int m_frameCount = 0;
        double m_timeAccum = 0.0;
        double m_lastPercent = 0.0;
        PipeClient pipe;
        std::string m_status;
        cocos2d::CCLabelBMFont* m_statusLabel = nullptr;
    };

    void update(float dt) {
        PlayerObject::update(dt);

        m_fields->m_frameCount++;
        m_fields->m_timeAccum += dt;

        // Sample observation every few frames to reduce I/O overhead while prototyping
        if (m_fields->m_frameCount % 5 == 0) {
            const auto obs = buildObservation(this, m_fields->m_timeAccum);
            const bool done = this->m_isDead || obs.levelPercent >= 100.f;
            double reward = (obs.levelPercent - m_fields->m_lastPercent) / 100.0;
            reward += 0.001; // small living reward
            if (done) reward -= 1.0; // penalty on death/end to shape
            m_fields->m_lastPercent = done ? 0.0 : obs.levelPercent;

            const std::string line = "OBS," + observationToCSV(obs) + "," + std::to_string(reward) + "," + (done ? "1" : "0");
            // lazy connect when first needed
            if (!m_fields->pipe.isConnected()) {
                m_fields->pipe.connect();
            }
            if (!m_fields->pipe.send(line)) {
                // fallback to file logging if pipe is down
                //writeLog(line);
            }
        }

        // Try to read action commands from the pipe (expects '0' or '1')
        std::string incoming;
        if (m_fields->pipe.tryReadLine(incoming)) {
            auto trim = [](std::string& s) {
                s.erase(std::remove_if(s.begin(), s.end(), [](unsigned char ch) { return ch == '\r' || ch == '\n' || ch == ' '; }), s.end());
            };

            std::stringstream ss(incoming);
            std::string line;
            while (std::getline(ss, line, '\n')) {
                trim(line);
                if (line.empty()) continue;
                if (line.rfind("STATUS,", 0) == 0) {
                    int gen = 0, indiv = 0;
                    double best = 0.0;
                    char comma;
                    std::stringstream ls(line.substr(7));
                    if (ls >> gen >> comma >> indiv >> comma >> best) {
                        m_fields->m_status = "Gen " + std::to_string(gen) + " Ind " + std::to_string(indiv) + " Best " + std::to_string(best);
                        if (!m_fields->m_statusLabel) {
                            if (auto* pl = PlayLayer::get()) {
                                auto* lbl = cocos2d::CCLabelBMFont::create("GA", "bigFont.fnt");
                                lbl->setScale(0.4f);
                                auto winSize = cocos2d::CCDirector::sharedDirector()->getWinSize();
                                lbl->setAnchorPoint({0.f, 1.f});
                                lbl->setPosition({10.f, winSize.height - 10.f});
                                pl->addChild(lbl, 1000);
                                m_fields->m_statusLabel = lbl;
                            }
                        }
                        if (m_fields->m_statusLabel) {
                            m_fields->m_statusLabel->setString(m_fields->m_status.c_str());
                        }
                    }
                } else if (line == "1") {
                    this->pushButton(PlayerButton::Jump);
                } else if (line == "0") {
                    this->releaseButton(PlayerButton::Jump);
                }
            }
        }
    }
};


