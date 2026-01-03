#include <Geode/Geode.hpp>
#include <Geode/modify/PlayLayer.hpp>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <limits>
#include <memory>
#include <vector>

using namespace geode; // NOLINT


struct Connection {
    std::vector<float> fromNeurons;
    std::vector<float> toNeurons;
    std::vector<std::vector<float>> weights;
    std::vector<float> biases;
};

class NeuralNetwork {
    std::vector<Connection> connections;
    public: 
    NeuralNetwork(const int* layerShape, size_t layerCount){
        if (!layerShape || layerCount < 2) {
            log::error("NeuralNetwork: invalid layer shape");
            return;
        }

        connections.reserve(layerCount - 1);
        for (size_t i = 0; i + 1 < layerCount; i++) {
            Connection conn;
            conn.fromNeurons.resize(static_cast<size_t>(layerShape[i]));
            conn.toNeurons.resize(static_cast<size_t>(layerShape[i + 1]));
            conn.biases.resize(static_cast<size_t>(layerShape[i + 1]), 0.0f);
            conn.weights.resize(static_cast<size_t>(layerShape[i + 1]), std::vector<float>(static_cast<size_t>(layerShape[i]), 0.0f));
            randomInitWeights(&conn.weights);
            randomInitBiases(&conn.biases);
            connections.push_back(std::move(conn));
        }
    }

    std::vector<float> forward(const std::vector<float>& input) {
        if (connections.empty()) {
            log::error("NeuralNetwork: forward called with no connections");
            return {};
        }

        if (input.size() != connections.front().fromNeurons.size()) {
            log::error("NeuralNetwork: input size {} does not match first layer {}", input.size(), connections.front().fromNeurons.size());
            return {};
        }

        std::vector<float> activations = input;
        for (const auto& conn : connections) {
            std::vector<float> newActivations(conn.toNeurons.size(), 0.0f);
            for (size_t j = 0; j < conn.toNeurons.size(); j++) {
                float sum = conn.biases[j];
                for (size_t i = 0; i < conn.fromNeurons.size(); i++) {
                    sum += activations[i] * conn.weights[j][i];
                }
                newActivations[j] = std::tanh(sum);
            }
            activations = newActivations;
        }
        return activations;
    }

    void mutate(float mutationChance) {
        for (auto& conn : connections) {
            for (auto& row : conn.weights) {
                for (auto& weight : row) {
                    if (static_cast<float>(rand()) / static_cast<float>(RAND_MAX) < mutationChance) {
                        weight += (static_cast<float>(rand()) / static_cast<float>(RAND_MAX) - 0.5f) * 0.2f; // small random change
                    }
                }
            }
            for (auto& bias : conn.biases) {
                if (static_cast<float>(rand()) / static_cast<float>(RAND_MAX) < mutationChance) {
                    bias += (static_cast<float>(rand()) / static_cast<float>(RAND_MAX) - 0.5f) * 0.2f; // small random change
                }
            }
        }
    }

    void copyFrom(NeuralNetwork* other) {
        if (!other) {
            log::error("NeuralNetwork: copyFrom called with null other");
            return;
        }
        connections = other->connections;
    }

    private:
    void randomInitWeights(std::vector<std::vector<float>>* weights) {
        for (auto& row : *weights) {
            for (auto& weight : row) {
                weight = (static_cast<float>(rand()) / RAND_MAX) * 2.f - 1.f; // Random float between -1 and 1
            }
        }
    }

    void randomInitBiases(std::vector<float>* biases) {
        for (auto& bias : *biases) {
            bias = (static_cast<float>(rand()) / RAND_MAX) * 2.f - 1.f; // Random float between -1 and 1
        }
    }
};


static int defaultLayerShape[] = {6,16,4};
static constexpr size_t defaultLayerCount = sizeof(defaultLayerShape) / sizeof(defaultLayerShape[0]);
static const int POPULATION_SIZE = 50;
static const int INJECT_COUNT = 5;
static const float MUTATION_RATE = 0.1f; // 10% mutation rate
static constexpr float ACTION_DURATIONS[4] = {0.0f, 0.08f, 0.18f, 0.35f};

struct $modify(DashAIPlayLayer, PlayLayer) {

    struct Fields{
        std::unique_ptr<NeuralNetwork> m_populationNetworks[POPULATION_SIZE];
        std::pair<NeuralNetwork*,float> distances[POPULATION_SIZE];
        float grades[POPULATION_SIZE] = {0.0f};
        int currentNetworkIdx = 0;
        bool isHolding = false;
        float holdTimer = 0.0f;
        float holdDuration = 0.0f;
        int jumpCount = 0;
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

    GameObject* nearestObstacle(PlayLayer* layer, const cocos2d::CCPoint& playerPos) {
        if (!layer || !layer->m_objects) return nullptr;
        GameObject* object = nullptr;
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
            object = obj;

        }
        return object;
    }

    int argmax(const std::vector<float>& v) {
        int best = 0;
        for (int i = 1; i < v.size(); i++) {
            if (v[i] > v[best]) best = i;
        }
        return best;
    }

    bool init(GJGameLevel* level,bool useReplay,bool dontCreateObjects) {
        if (!PlayLayer::init(level,useReplay,dontCreateObjects)) {
            log::error("DashAI: failed to initialize PlayLayer");
            return false;
        }

        static bool seeded = false;
        if (!seeded) {
            srand(static_cast<unsigned>(time(nullptr)));
            seeded = true;
        }

        for(int i = 0; i < POPULATION_SIZE; i++) {
            m_fields->m_populationNetworks[i] = std::make_unique<NeuralNetwork>(defaultLayerShape, defaultLayerCount);
        }

        
        return true;
    }

    std::vector<int> getTopElitesIndices(const float* grades, size_t eliteCount) {
        std::vector<std::pair<float, int>> gradeIdxPairs;
        gradeIdxPairs.reserve(POPULATION_SIZE);
        for (int i = 0; i < POPULATION_SIZE; i++) {
            gradeIdxPairs.emplace_back(grades[i], i);
        }
        std::sort(gradeIdxPairs.begin(), gradeIdxPairs.end(), std::greater<>());
        std::vector<int> eliteIndices;
        eliteIndices.reserve(eliteCount);
        for (size_t i = 0; i < eliteCount && i < gradeIdxPairs.size(); i++) {
            eliteIndices.push_back(gradeIdxPairs[i].second);
        }
        return eliteIndices;
    }

    void createNextGeneration(const std::vector<int>& eliteIndices) {
        if (eliteIndices.empty()) {
            log::error("DashAI: no elites available for next generation");
            return;
        }

        std::vector<NeuralNetwork> eliteCopies;
        eliteCopies.reserve(eliteIndices.size());
        for (size_t i = 0; i < eliteIndices.size(); i++) {
            auto* elitePtr = m_fields->m_populationNetworks[eliteIndices[i]].get();
            if (elitePtr) {
                eliteCopies.push_back(*elitePtr);
            }
        }

        if (eliteCopies.empty()) {
            log::error("DashAI: failed to clone elites for next generation");
            return;
        }

        // Keep best elite unmutated in slot 0.
        m_fields->m_populationNetworks[0] = std::make_unique<NeuralNetwork>(defaultLayerShape, defaultLayerCount);
        m_fields->m_populationNetworks[0]->copyFrom(&eliteCopies[0]);

        // Fill the rest with mutated copies cycling through elites.
        for (int i = 1; i < POPULATION_SIZE; i++) {
            size_t parentIdx = static_cast<size_t>(i) % eliteCopies.size();
            m_fields->m_populationNetworks[i] = std::make_unique<NeuralNetwork>(defaultLayerShape, defaultLayerCount);
            if(POPULATION_SIZE - i > INJECT_COUNT) {
                m_fields->m_populationNetworks[i]->copyFrom(&eliteCopies[parentIdx]);
                m_fields->m_populationNetworks[i]->mutate(MUTATION_RATE); // Mutate with 10% chance
            }
        }
    }

    float getBestDistance(const std::pair<NeuralNetwork*,float>* distances, size_t count) {
        float bestDistance = 0.0f;
        for (size_t i = 0; i < count; i++) {
            if (distances[i].second > bestDistance) {
                bestDistance = distances[i].second;
            }
        }
        return bestDistance;
    }

    float getAverageDistance(const std::pair<NeuralNetwork*,float>* distances, size_t count) {
        float totalDistance = 0.0f;
        for (size_t i = 0; i < count; i++) {
            totalDistance += distances[i].second;
        }
        return totalDistance / static_cast<float>(count);
    }

    void postUpdate(float dt) {
        PlayLayer::postUpdate(dt);
        std::vector<float> gameData{};
        GameObject* nearestObs = nearestObstacle(this, this->m_player1->getPosition());
        if(!nearestObs) {
            log::error("DashAI: no nearest obstacle found");
            return;
        }
        gameData.resize(defaultLayerShape[0], 0.0f);
        gameData[0] = this->m_player1->getPosition().y / 300.0f;
        gameData[1] = this->m_player1->m_yVelocity / 60.0f;
        gameData[2] = nearestObs->getPosition().x / 300.0f;
        gameData[3] = nearestObs->m_height / 100.0f;
        gameData[4] = m_player1->m_isOnGround ? 1.0f : 0.0f;
        gameData[5] = m_player1->m_isShip ? 1.0f : 0.0f;
        auto output = m_fields->m_populationNetworks[m_fields->currentNetworkIdx]->forward(gameData);
        if (output.empty()) {
            return;
        }
        auto action = argmax(output);

        if (action == 0) {
            if (m_fields->isHolding) {
                m_player1->releaseButton(PlayerButton::Jump);
                m_fields->isHolding = false;
                m_fields->holdTimer = 0.0f;
                m_fields->holdDuration = 0.0f;
            }
        } else {
            if (!m_fields->isHolding) {
                size_t idx = static_cast<size_t>(std::min(action, 3));
                m_fields->holdDuration = ACTION_DURATIONS[idx];
                m_fields->holdTimer = 0.0f;
                m_player1->pushButton(PlayerButton::Jump);
                m_fields->isHolding = true;
                m_fields->jumpCount++;
            }
        }

        if (m_fields->isHolding) {
            m_fields->holdTimer += dt;
            if (m_fields->holdTimer >= m_fields->holdDuration) {
                m_player1->releaseButton(PlayerButton::Jump);
                m_fields->isHolding = false;
                m_fields->holdTimer = 0.0f;
                m_fields->holdDuration = 0.0f;
            }
        }
        if(m_player1->m_isDead) {
            m_fields->grades[m_fields->currentNetworkIdx] = this->m_player1->getPosition().x - m_fields->jumpCount; // Use distance traveled as grade for now
            m_fields->distances[m_fields->currentNetworkIdx] = {m_fields->m_populationNetworks[m_fields->currentNetworkIdx].get(), (this->m_player1->getPosition().x / this->m_levelLength) * 100.0f};
            m_fields->currentNetworkIdx = (m_fields->currentNetworkIdx + 1) % POPULATION_SIZE;
            if(m_fields->currentNetworkIdx == 0) {
                m_fields->jumpCount = 0;
                float bestDistance = getBestDistance(m_fields->distances, POPULATION_SIZE);
                float avgDistance = getAverageDistance(m_fields->distances, POPULATION_SIZE);
                log::info("DashAI: Generation completed. Best Distance: {:.2f}, Average Distance: {:.2f}", bestDistance, avgDistance);
                auto elitesIdx = getTopElitesIndices(m_fields->grades, 5);
                createNextGeneration(elitesIdx);
                log::info("DashAI: Created next generation of neural networks");
            }
            this->resetLevel();
        }
    }
}; 