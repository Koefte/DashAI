#include <Geode/Geode.hpp>
#include <Geode/modify/PlayLayer.hpp>
#include <atomic>
#include <chrono>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <string>
#include <string_view>
#include <thread>
#include <sstream>
#include <limits>
#include <cstdlib>

#if defined(_WIN32)
#define NOMINMAX
#include <windows.h>
#endif

using namespace geode::prelude;

namespace dashai {

#if defined(_WIN32)
class PipeServer {
public:
    void start() {
        bool expected = false;
        if (!m_started.compare_exchange_strong(expected, true)) return;
        m_thread = std::thread(&PipeServer::serverLoop, this);
    }

    void stop() {
        m_stop.store(true);
        m_cv.notify_all();
        if (m_thread.joinable()) m_thread.join();
    }

    bool connected() const { return m_connected.load(); }

    bool tryPop(std::string& out) {
        std::scoped_lock lock(m_queueMutex);
        if (m_inbox.empty()) return false;
        out = std::move(m_inbox.front());
        m_inbox.pop();
        return true;
    }

    void enqueueLine(std::string_view payload) {
        if (!connected()) return;
        std::string line(payload);
        line.push_back('\n');
        {
            std::lock_guard lock(m_outMutex);
            m_outbox.push(std::move(line));
        }
        m_cv.notify_one();
    }

private:
    void serverLoop() {
        while (!m_stop.load()) {
            HANDLE pipe = CreateNamedPipeA(
                "\\\\.\\pipe\\DashAI",
                PIPE_ACCESS_DUPLEX,
                PIPE_TYPE_MESSAGE | PIPE_READMODE_MESSAGE | PIPE_WAIT,
                1,
                4096,
                4096,
                0,
                nullptr
            );
            if (pipe == INVALID_HANDLE_VALUE) {
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
                continue;
            }

            BOOL ok = ConnectNamedPipe(pipe, nullptr);
            if (!ok && GetLastError() != ERROR_PIPE_CONNECTED) {
                CloseHandle(pipe);
                continue;
            }

            m_pipe = pipe;
            m_connected.store(true);
            m_pipeAlive.store(true);
            log::info("DashAI pipe client connected");
            std::thread writer(&PipeServer::writeLoop, this, pipe);
            readLoop(pipe);
            m_pipeAlive.store(false);
            m_cv.notify_all();
            if (writer.joinable()) writer.join();
            m_connected.store(false);
            log::info("DashAI pipe client disconnected");
            CloseHandle(pipe);
            m_pipe = INVALID_HANDLE_VALUE;
        }
    }

    void readLoop(HANDLE pipe) {
        std::string buffer;
        buffer.reserve(2048);
        char chunk[512];
        while (!m_stop.load()) {
            DWORD read = 0;
            BOOL ok = ReadFile(pipe, chunk, sizeof(chunk), &read, nullptr);
            if (!ok || read == 0) break;
            buffer.append(chunk, chunk + read);
            std::size_t pos = 0;
            while ((pos = buffer.find('\n')) != std::string::npos) {
                std::string line = buffer.substr(0, pos);
                buffer.erase(0, pos + 1);
                if (!line.empty()) {
                    std::scoped_lock lock(m_queueMutex);
                    m_inbox.push(std::move(line));
                }
            }
        }
    }

    void writeLoop(HANDLE pipe) {
        while (!m_stop.load() && m_pipeAlive.load()) {
            std::string line;
            {
                std::unique_lock lock(m_outMutex);
                m_cv.wait(lock, [&]{ return m_stop.load() || !m_pipeAlive.load() || !m_outbox.empty(); });
                if (m_stop.load() || !m_pipeAlive.load()) break;
                line = std::move(m_outbox.front());
                m_outbox.pop();
            }
            DWORD written = 0;
            BOOL ok = WriteFile(pipe, line.data(), static_cast<DWORD>(line.size()), &written, nullptr);
            if (!ok) break;
            FlushFileBuffers(pipe);
        }
        // clear leftover messages when pipe drops
        std::lock_guard lock(m_outMutex);
        std::queue<std::string> empty;
        std::swap(m_outbox, empty);
    }

    std::atomic<bool> m_started{false};
    std::atomic<bool> m_stop{false};
    std::atomic<bool> m_connected{false};
    std::atomic<bool> m_pipeAlive{false};
    std::thread m_thread;
    HANDLE m_pipe = INVALID_HANDLE_VALUE;
    std::mutex m_queueMutex;
    std::queue<std::string> m_inbox;
    std::mutex m_outMutex;
    std::condition_variable m_cv;
    std::queue<std::string> m_outbox;
};
#else
class PipeServer {
public:
    void start() {}
    void stop() {}
    bool connected() const { return false; }
    bool tryPop(std::string&) { return false; }
    void sendLine(std::string_view) {}
};
#endif

class AIBridge {
public:
    static AIBridge& get() {
        static AIBridge inst;
        return inst;
    }

    void start() { m_pipe.start(); }
    void shutdown() { m_pipe.stop(); }

    void onLevelStart(PlayLayer* layer) {
        m_layer.store(layer);
        m_attempt.fetch_add(1);
        m_lastPercent = 0.0f;
        
        // Create stats label if it doesn't exist
        if (!m_statsLabel) {
            m_statsLabel = cocos2d::CCLabelBMFont::create("Best: 0.0% | Avg: 0.0%", "bigFont.fnt");
            m_statsLabel->setScale(0.5f);
            m_statsLabel->setZOrder(1000);
            m_statsLabel->retain();
        }
        
        // Position label at top of screen
        auto winSize = cocos2d::CCDirector::sharedDirector()->getWinSize();
        m_statsLabel->setPosition({winSize.width / 2, winSize.height - 20.f});
        layer->addChild(m_statsLabel);
    }

    void onLevelStop(PlayLayer* layer) {
        // Track this run's percentage
        if (m_lastPercent > 0.0f) {
            m_runPercents.push_back(m_lastPercent);
            // Keep only last 100 runs
            if (m_runPercents.size() > 100) {
                m_runPercents.erase(m_runPercents.begin());
            }
            // Update best
            if (m_lastPercent > m_bestPercent) {
                m_bestPercent = m_lastPercent;
            }
        }
        
        // Remove label from layer
        if (m_statsLabel && m_statsLabel->getParent() == layer) {
            m_statsLabel->removeFromParent();
        }
        
        PlayLayer* expected = layer;
        m_layer.compare_exchange_strong(expected, nullptr);
    }

    void setSpeedMultiplier(float multiplier) {
        m_speedMultiplier = std::max(0.1f, std::min(10.0f, multiplier));
        auto scheduler = cocos2d::CCDirector::sharedDirector()->getScheduler();
        if (scheduler) {
            scheduler->setTimeScale(m_speedMultiplier);
        }
        if (m_debug) {
            log::info("DashAI speed multiplier set to {:.2f}x", m_speedMultiplier);
        }
    }

    float getSpeedMultiplier() const {
        return m_speedMultiplier;
    }

    void onPostUpdate(PlayLayer* layer, float) {
        if (layer != m_layer.load()) return;
        handleCommands(layer);
        maybeSendState(layer);
    }

private:
    AIBridge() {
        m_debug = std::getenv("DASHAI_DEBUG") != nullptr;
        if (m_debug) {
            log::info("DashAI debug logging enabled (env DASHAI_DEBUG)");
        }
    }
    ~AIBridge() { shutdown(); }

    void handleCommands(PlayLayer* layer) {
        std::string cmd;
        while (m_pipe.tryPop(cmd)) {
            applyCommand(layer, cmd);
        }
    }

    bool parseFlag(std::string_view cmd, std::string_view key) {
        auto pos = cmd.find(key);
        if (pos == std::string::npos) return false;
        pos += key.size();
        if (pos >= cmd.size() || cmd[pos] != '=') return false;
        pos += 1;
        if (pos >= cmd.size()) return false;
        char c = static_cast<char>(cmd[pos]);
        return c == '1' || c == 't' || c == 'T' || c == 'y' || c == 'Y';
    }

    struct ObstacleInfo {
        float x = -1.f;
        float y = -1.f;
        float w = -1.f;
        float h = -1.f;
        bool found = false;
    };

    bool isObstacle(GameObject* obj) {
        if (!obj) return false;
        auto type = obj->getType();
        switch (type) {
            case GameObjectType::Hazard:
            case GameObjectType::AnimatedHazard:
            case GameObjectType::CollisionObject:
            case GameObjectType::Solid:
            case GameObjectType::Slope:
            case GameObjectType::Breakable:
                return true;
            default:
                return false;
        }
    }

    ObstacleInfo nearestObstacle(PlayLayer* layer, const cocos2d::CCPoint& playerPos) {
        ObstacleInfo info;
        if (!layer || !layer->m_objects) return info;
        float bestDx = std::numeric_limits<float>::infinity();
        cocos2d::CCObject* raw = nullptr;
        CCARRAY_FOREACH(layer->m_objects, raw) {
            auto obj = static_cast<GameObject*>(raw);
            if (!isObstacle(obj)) continue;
            auto rect = obj->getObjectRect();
            float left = rect.origin.x;
            float dx = left - playerPos.x;
            if (dx < 0.f) continue;
            if (dx >= bestDx) continue;
            bestDx = dx;
            info.x = rect.origin.x;
            info.y = rect.origin.y;
            info.w = rect.size.width;
            info.h = rect.size.height;
            info.found = true;
        }
        return info;
    }

    void applyCommand(PlayLayer* layer, std::string_view cmd) {
        // Handle speed command
        if (cmd.rfind("speed", 0) == 0) {
            auto pos = cmd.find("=");
            if (pos != std::string::npos) {
                try {
                    float speed = std::stof(std::string(cmd.substr(pos + 1)));
                    setSpeedMultiplier(speed);
                } catch (...) {
                    if (m_debug) {
                        log::debug("Invalid speed value in command");
                    }
                }
            }
            return;
        }
        
        if (cmd.rfind("action", 0) != 0) return;
        auto player = layer->m_player1;
        if (!player) return;
        bool jump = parseFlag(cmd, "jump");
        bool hold = parseFlag(cmd, "hold");
        if (m_debug) {
            log::debug("cmd action jump={} hold={}", jump, hold);
        }
        if (jump || hold) {
            player->pushButton(PlayerButton::Jump);
        } else {
            player->releaseButton(PlayerButton::Jump);
        }
    }

    void maybeSendState(PlayLayer* layer) {
        if (!m_pipe.connected()) return;
        auto now = std::chrono::steady_clock::now();
        if (now - m_lastSend < std::chrono::milliseconds(33)) return;
        m_lastSend = now;

        auto player = layer->m_player1;
        if (!player) return;

        cocos2d::CCPoint pos = player->getPosition();
        float vy = player->m_yVelocity;
        float percent = layer->getCurrentPercent();
        bool alive = !player->m_isDead;
        auto obstacle = nearestObstacle(layer, pos);
        
        // Update last percent for tracking
        m_lastPercent = percent;
        
        // Update best percent
        if (percent > m_bestPercent) {
            m_bestPercent = percent;
        }
        
        // Calculate average percentage
        float avgPercent = 0.0f;
        if (!m_runPercents.empty()) {
            float sum = 0.0f;
            for (float p : m_runPercents) {
                sum += p;
            }
            avgPercent = sum / m_runPercents.size();
        }
        
        // Update stats label
        if (m_statsLabel) {
            char labelText[128];
            snprintf(labelText, sizeof(labelText), "Best: %.1f%% | Avg: %.1f%% | Runs: %zu", 
                     m_bestPercent, avgPercent, m_runPercents.size());
            m_statsLabel->setString(labelText);
        }

        std::ostringstream oss;
        oss.setf(std::ios::fixed);
        oss.precision(3);
        oss << "state "
            << "attempt=" << m_attempt.load() << ' '
            << "percent=" << percent << ' '
            << "x=" << pos.x << ' '
            << "y=" << pos.y << ' '
            << "vy=" << vy << ' '
            << "alive=" << (alive ? 1 : 0) << ' '
            << "ob_x=" << obstacle.x << ' '
            << "ob_y=" << obstacle.y << ' '
            << "ob_w=" << obstacle.w << ' '
            << "ob_h=" << obstacle.h << ' '
            << "speed=" << m_speedMultiplier;
        m_pipe.enqueueLine(oss.str());

        if (m_debug) {
            auto nowDebug = std::chrono::steady_clock::now();
            if (nowDebug - m_lastDebugLog > std::chrono::milliseconds(1000)) {
                m_lastDebugLog = nowDebug;
                log::debug("state pct={:.2f} x={:.1f} y={:.1f} vy={:.2f} ob=({}, {}, {}, {}) alive={}",
                    percent, pos.x, pos.y, vy,
                    obstacle.x, obstacle.y, obstacle.w, obstacle.h,
                    alive);
            }
        }
    }

    PipeServer m_pipe;
    std::atomic<PlayLayer*> m_layer{nullptr};
    std::atomic<int> m_attempt{0};
    std::chrono::steady_clock::time_point m_lastSend{};
    bool m_debug = false;
    std::chrono::steady_clock::time_point m_lastDebugLog{};
    float m_speedMultiplier = 1.0f;
    float m_bestPercent = 0.0f;
    float m_lastPercent = 0.0f;
    std::vector<float> m_runPercents;
    cocos2d::CCLabelBMFont* m_statsLabel = nullptr;
};

} // namespace dashai

$on_mod(Loaded) {
    dashai::AIBridge::get().start();
    log::info("DashAI pipe server ready at \\\\.\\\\pipe\\\\DashAI");
}

class $modify(DashAIPlayLayer, PlayLayer) {
public:
    bool init(GJGameLevel* level, bool useReplay, bool dontCreateObjects) {
        if (!PlayLayer::init(level, useReplay, dontCreateObjects)) return false;
        dashai::AIBridge::get().onLevelStart(this);
        return true;
    }

    void onExit() {
        dashai::AIBridge::get().onLevelStop(this);
        PlayLayer::onExit();
    }

    void postUpdate(float dt) {
        dashai::AIBridge::get().onPostUpdate(this, dt);
        PlayLayer::postUpdate(dt);
    }
};
