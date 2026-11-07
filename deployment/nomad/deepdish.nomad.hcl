job "wer7892-diego-hw2" {
  datacenters = ["dc1"]
  type = "batch"
  meta {
    owner = "wer7892"
  }
  group "single-gpu" {
    count = 1
    restart {
      attempts = 0
      mode = "fail"
    }
    reschedule {
      attempts = 0
      unlimited = false
    }
    task "dqn" {
      driver = "docker"
      user = "1215055794:1215055794"
      config {
        image = "pytorch/pytorch:2.5.1-cuda11.8-cudnn9-runtime"
        command = "/bin/bash"
        args = ["/workspace/deployment/nomad/run_nomad.sh"]
        work_dir = "/workspace"
        volumes = ["/nfs/home/wer7892/diego_hw2_codex_20261106:/workspace"]
      }
      env {
        HOME = "/workspace"
        DEEPDISH_NODE = "${node.unique.name}"
      }
      resources {
        cpu = 2000
        memory = 8192
        device "nvidia/gpu" {
          count = 1
        }
      }
      logs {
        max_files = 5
        max_file_size = 10
      }
    }
  }
}
