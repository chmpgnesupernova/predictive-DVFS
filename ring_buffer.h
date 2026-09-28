#ifndef RING_BUFFER_H
#define RING_BUFFER_H

#include <array>
#include <cstddef>
#include <mutex>
#include <condition_variable>

/*
 * 고정 크기 thread-safe ring buffer (Vision -> LLM)
 *  - push: 가득 차면 가장 오래된 항목을 덮어쓴다 (로봇에서는 최신 인식 결과가 더 중요)
 *  - pop : 항목이 생길 때까지 block. close() 후 버퍼가 비면 false 반환
 */
enum class PushResult {
    Accepted,     // 새 항목 추가
    Overwrote,    // 가득 차서 가장 오래된 항목을 덮어씀 (대기 개수 변화 없음)
    Closed        // 이미 close 됨, 무시
};

template <typename T, std::size_t N>
class RingBuffer {
    static_assert(N > 0, "RingBuffer capacity must be > 0");

public:
    PushResult push(const T& item) {
        PushResult result;
        {
            std::lock_guard<std::mutex> lock(mtx_);
            if (closed_) return PushResult::Closed;

            if (count_ == N) {
                // full: 가장 오래된 항목(head) 을 버리고 그 자리에 기록
                buf_[head_] = item;
                head_ = (head_ + 1) % N;
                ++dropped_;
                result = PushResult::Overwrote;
            } else {
                buf_[(head_ + count_) % N] = item;
                ++count_;
                result = PushResult::Accepted;
            }
        }
        cv_.notify_one();
        return result;
    }

    bool pop(T& out) {
        std::unique_lock<std::mutex> lock(mtx_);
        cv_.wait(lock, [this] { return count_ > 0 || closed_; });
        if (count_ == 0) return false;   // closed && empty

        out = buf_[head_];
        head_ = (head_ + 1) % N;
        --count_;
        return true;
    }

    void close() {
        {
            std::lock_guard<std::mutex> lock(mtx_);
            closed_ = true;
        }
        cv_.notify_all();
    }

    std::size_t size() {
        std::lock_guard<std::mutex> lock(mtx_);
        return count_;
    }

    std::size_t dropped() {
        std::lock_guard<std::mutex> lock(mtx_);
        return dropped_;
    }

private:
    std::array<T, N> buf_{};
    std::size_t head_ = 0;
    std::size_t count_ = 0;
    std::size_t dropped_ = 0;
    bool closed_ = false;
    std::mutex mtx_;
    std::condition_variable cv_;
};

#endif // RING_BUFFER_H
