#ifndef PREDICTIVE_DVFS_CLASS_H
#define PREDICTIVE_DVFS_CLASS_H

#include <iostream>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <string>

// --- FSM State 및 Predictive DVFS Controller ---
enum FSMState {
    STRONGLY_NOT_TAKEN,
    WEAKLY_NOT_TAKEN,
    WEAKLY_TAKEN,
    STRONGLY_TAKEN
};

class PredictiveDVFS {
private:
    FSMState state = WEAKLY_TAKEN;
    std::mutex mtx;
    bool early_scale = false;

    void update(bool is_hit) {
        if (is_hit) {
            switch (state) {
                case STRONGLY_NOT_TAKEN: state = WEAKLY_NOT_TAKEN; break;
                case WEAKLY_NOT_TAKEN:   state = WEAKLY_TAKEN; break;
                case WEAKLY_TAKEN:       state = STRONGLY_TAKEN; break;
                case STRONGLY_TAKEN:     state = STRONGLY_TAKEN; break;
            }
            std::cout << "[FSM] Update: Hit (LLM Request) -> New State: " << stateToString(state) << "\n";
        } else {
            switch (state) {
                case STRONGLY_TAKEN:     state = WEAKLY_TAKEN; break;
                case WEAKLY_TAKEN:       state = WEAKLY_NOT_TAKEN; break;
                case WEAKLY_NOT_TAKEN:   state = STRONGLY_NOT_TAKEN; break;
                case STRONGLY_TAKEN:     state = STRONGLY_NOT_TAKEN; break;
            }
            std::cout << "[FSM] Update: Miss (No Request) -> New State: " << stateToString(state) << "\n";
        }
    }

    std::string stateToString(FSMState s) {
        switch (s) {
            case STRONGLY_NOT_TAKEN: return "Strongly Not Taken";
            case WEAKLY_NOT_TAKEN:   return "Weakly Not Taken";
            case WEAKLY_TAKEN:       return "Weakly Taken";
            case STRONGLY_TAKEN:     return "Strongly Taken";
            default: return "Unknown";
        }
    }

public:
    void scaleFrequencyEarly() {
        std::lock_guard<std::mutex> lock(mtx);
        if (state == STRONGLY_TAKEN || state == WEAKLY_TAKEN) {
            early_scale = true;
            std::cout << "[DVFS] Early Scaling Triggered (Predictive). Scaling asynchronously...\n";
        } else {
            early_scale = false;
        }
    }
    
    void scaleFrequencyReactive(bool is_hit) {
        std::lock_guard<std::mutex> lock(mtx);
        
        if (!early_scale && is_hit) {
            std::cout << "[DVFS] Reactive Scaling Triggered. 15ms overhead penalty applied!\n";
            std::this_thread::sleep_for(std::chrono::milliseconds(15)); 
            std::cout << "[DVFS] GPU Frequency is now HIGH.\n";
        } else if (early_scale && is_hit) {
            std::cout << "[DVFS] GPU Frequency is already HIGH (15ms Latency Hidden).\n";
        } else if (early_scale && !is_hit) {
            std::cout << "[DVFS] False Alarm: Scaled but no LLM request (Energy penalty).\n";
        }
        
        update(is_hit);
    }
};

#endif
