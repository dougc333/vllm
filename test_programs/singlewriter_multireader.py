# the singel writer multi reader test program
# The problem is performacne 
# using locks is slow
# allocating memory is slow
# garbage collection is slow. 
# Use ring buffer to keep memory usage bounded. 

# 5 metrics 
# 1) ringbuffer_writer_seq_total: counter
# 2) ringbuffer_reader_seq_total: counter 
# 3)ringbuffer_capacity_chunks: guage
# 4) ringbuffer_writer_stall_seconds: histogram
# 5) ringbuffer_reader_wait_seconds{reader_id="X"}: histogram
# 6) ringbuffer_overflow_events_total: counter
#
# cant look at sequnece nubmers since they grow infiinitely
# occupancy = writer_seq - min(readerseq_0, readerseq_1, readerseq_2, readerseq_N)
# if occupancy > capacity then writer is blocked
# the slowest reader is the lowest reader_seq 
#
from prometheus_client import Counter, Histogram, Gauge
import time

# 1. Define Metrics
WRITER_SEQ = Counter('ringbuffer_writer_seq_total', 'Total chunks written')
READER_SEQ = Counter('ringbuffer_reader_seq_total', 'Total chunks read', ['reader_id'])
WRITER_STALL = Histogram('ringbuffer_writer_stall_seconds', 'Time writer spent blocked')
READER_WAIT = Histogram('ringbuffer_reader_wait_seconds', 'Time reader spent blocked', ['reader_id'])

class InstrumentedShmRingBuffer:
    def __init__(self, max_chunks, n_readers):
        self.max_chunks = max_chunks
        self.n_readers = n_readers
        self.writer_seq = 0
        self.reader_seqs = [0] * n_readers

    @contextmanager
    def acquire_write(self):
        start_wait = time.monotonic()
        
        # --- THE HOT LOOP (Where the writer blocks) ---
        while True:
            slowest_reader_seq = min(self.reader_seqs)
            if self.writer_seq - slowest_reader_seq < self.max_chunks:
                break # We have space! Exit the loop.
            
            # If we are here, the buffer is full. We are stalling.
            sched_yield() # Yield CPU

        # Record how long we spent blocked waiting for readers
        stall_duration = time.monotonic() - start_wait
        WRITER_STALL.observe(stall_duration)

        # Yield the buffer to the caller to write data...
        yield self.get_data_buffer(self.writer_seq % self.max_chunks)

        # Update sequences AFTER write is complete
        self.writer_seq += 1
        WRITER_SEQ.inc()

    @contextmanager
    def acquire_read(self, reader_id):
        start_wait = time.monotonic()
        
        # --- THE HOT LOOP (Where the reader blocks) ---
        while True:
            if self.reader_seqs[reader_id] < self.writer_seq:
                break # There is data! Exit the loop.
            
            # If we are here, buffer is empty. Reader is starving.
            self.spin_condition.wait(timeout_ms=1) 

        # Record how long we spent blocked waiting for the writer
        wait_duration = time.monotonic() - start_wait
        READER_WAIT.labels(reader_id=reader_id).observe(wait_duration)

        # Yield the buffer to the caller to read data...
        yield self.get_data_buffer(self.reader_seqs[reader_id] % self.max_chunks)

        # Update sequences AFTER read is complete
        self.reader_seqs[reader_id] += 1
        READER_SEQ.labels(reader_id=reader_id).inc()