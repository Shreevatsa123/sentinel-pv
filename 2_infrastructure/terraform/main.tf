# 2_infrastructure/terraform/main.tf

# CONTEXT: This file automates the creation of the cloud server (VM).
# It sets up the firewall, creates the machine, and auto-installs Docker + n8n.

provider "google" {
  # I am using a variable here so your actual Project ID isn't hardcoded in public code.
  project = var.project_id
  region  = "us-east1"
  zone    = "us-east1-b"
}

# 1. The Firewall (Security Guard)
resource "google_compute_firewall" "sentinel_rules" {
  name    = "sentinel-allow-n8n-streamlit"
  network = "default"

  allow {
    protocol = "tcp"
    # Port 22:   For SSH access (Command Line)
    # Port 5678: For n8n Workflow Editor
    # Port 8501: For Streamlit Dashboard (Frontend)
    ports    = ["22", "5678", "8501"]
  }

  source_ranges = ["0.0.0.0/0"] # Open to the world (For development convenience)
  target_tags   = ["sentinel-node"]
}

# 2. The Virtual Machine (The Body)
# This is the actual server where the AI Agent runs.
resource "google_compute_instance" "sentinel_vm" {
  name         = "sentinel-core"
  machine_type = "e2-micro"  # Chosen because it is Google Cloud Free Tier eligible
  tags         = ["sentinel-node"]

  boot_disk {
    initialize_params {
      image = "ubuntu-os-cloud/ubuntu-2204-lts"
      size  = 30 # Max size allowed in Free Tier
    }
  }

  network_interface {
    network = "default"
    access_config {
      # This block assigns a Public IP so we can access n8n from our browser
    }
  }

  # 3. Startup Script (The Brain Transplant)
  # This runs automatically when the VM turns on. It installs Docker
  # and immediately starts the n8n orchestrator so we don't have to do it manually.
  metadata_startup_script = <<-EOF
    #!/bin/bash
    apt-get update
    apt-get install -y docker.io
    systemctl start docker
    systemctl enable docker
    
    # Start n8n (The Workflow Orchestrator)
    docker run -d \
      --name n8n \
      -p 5678:5678 \
      -e GENERIC_TIMEZONE="America/New_York" \
      -v n8n_data:/home/node/.n8n \
      docker.n8n.io/n8nio/n8n
  EOF
}

# 4. Output
# Prints the IP address in the terminal so you can copy-paste it immediately.
output "external_ip" {
  value = google_compute_instance.sentinel_vm.network_interface[0].access_config[0].nat_ip
}