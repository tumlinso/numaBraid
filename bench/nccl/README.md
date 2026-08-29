# NCCL send/receive characterization

`nccl_sendrecv_benchmark.cu` initializes all four validated physical roles in
one NCCL communicator and measures directed `ncclSend`/`ncclRecv` pairs. Roles
are resolved from PCI identities. Each transfer is byte-validated, warmed up,
and reported as JSON Lines with repeated-run statistics.

`--nvb-disable 0` explicitly permits NCCL's default intermediate-GPU NVB
mechanism. `--nvb-disable 1` is the controlled comparison that disables it.
The benchmark sets debug and topology-dump variables only in its own process;
it does not modify the caller's shell environment.

`--p2p-level PHB` is the narrow controlled specialization tested after the
default run. With NVB enabled, NCCL classifies an intermediate-GPU diagonal as
PHB; with NVB disabled, the same physical diagonal remains SYS.
