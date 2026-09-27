#ifndef PREDICTIVE_DVFS_CLASS_H
#define PREDICTIVE_DVFS_CLASS_H

#include <iostream>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <string>

/*
 * class 에 1) hit_count, 2) miss_count 를 도입
 * 
 * 1) TAKEN state 의 경우:
 *  miss_count < 20 일 경우 stay,
 *  miss_count >= 20 일 경우 switch
 *
 * 2) NOT_TAKEN state 의 경우:
 *  hit_count < 2 일 경우 stay,
 *  hit_count >= 일 경우 switch
 *  */


// State 는 2 state 로 변경
enum FSMState {
    TAKEN,
    NOT_TAKEN
};

class PredictiveDVFS {
private:
    FSMState state = TAKEN;
    std::mutex mtx;
    bool early_scale = false;

    int miss_count = 0;
    int hit_count = 0;

    void update(bool is_hit) {
        if (is_hit) {
            switch (state) {
                case TAKEN:     miss_count = 0; state = TAKEN; break;
                case NOT_TAKEN: (++hit_count < 2) ? state = NOT_TAKEN : state = TAKEN; break;
            }
        } else {
            switch (state) {
                case TAKEN:     (++miss_count < 20) ? state = TAKEN : state = NOT_TAKEN; break;
                case NOT_TAKEN: hit_count = 0; state = NOT_TAKEN; break;
            }
        }
    }

    std::string stateToString(FSMState s) {
        switch (s) {
            case TAKEN:         return "TAKEN";
            case NOT_TAKEN:     return "NOT_TAKEN";
            default:            return "UNKNOWN";
        }
    }

public:
    void scaleFrequencyEarly() {
        std::lock_guard<std::mutex> lock(mtx);
        if (state == TAKEN) {
            // DVFS start, scale freq
            std::string cmd = "sudo echo userspace > /sys/class/devfreq/170000.gpu/governor \
                               sudo echo 624750000 > /sys/calss/devfreq/170000.gpu/userspace/set_freq";
            int ret = system(cmd.c_str());
            (void)ret;
            early_scale = true;

        } else {
            // already low freq
            early_scale = false;
        }
    }
    
    void scaleFrequencyReactive(bool is_hit) {
        std::lock_guard<std::mutex> lock(mtx);
        
        if (!early_scale && is_hit) {
            // DVFS start, scale freq
            std::string cmd = "sudo echo userspace > /sys/class/devfreq/170000.gpu/governor \
                               sudo echo 624750000 > /sys/calss/devfreq/170000.gpu/userspace/set_freq";
            int ret = system(cmd.c_str());
            (void)ret;

        } else if (early_scale && !is_hit) {
            // DVFS start, False alram, down freq
            std::string cmd = "sudo echo userspace > /sys/class/devfreq/170000.gpu/governor \
                               sudo echo 306000000 > /sys/calss/devfreq/170000.gpu/userspace/set_freq";
            int ret = system(cmd.c_str());
            (void)ret;
        }
        // early_scale && is_hit 의 경우와 !early_scale && !is_hit 경우는 skip.

        update(is_hit);
    }
};

#endif
