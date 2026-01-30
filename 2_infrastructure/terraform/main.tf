# 2_infrastructure/terraform/main.tf

provider "google" {
  project = "neuralops" # Matches your GCP Project ID
  region  = "us-east1"
  zone    = "us-east1-b"
}

# 1. The Firewall (Security Guard)
resource "google_compute_firewall" "sentinel_rules" {
  name    = "sentinel-allow-n8n-streamlit"
  network = "default"

  allow {
    protocol = "tcp"
    # Added 8501 for Streamlit Dashboard
    ports    = ["22", "5678", "8501"] 
  }

  source_ranges = ["0.0.0.0/0"]
  target_tags   = ["sentinel-node"]
}

# 2. The Virtual Machine (The Body)
resource "google_compute_instance" "sentinel_vm" {
  name         = "sentinel-core"
  machine_type = "e2-micro"  # Free Tier Eligible
  tags         = ["sentinel-node"]

  boot_disk {
    initialize_params {
      image = "ubuntu-os-cloud/ubuntu-2204-lts"
      size  = 30 # Max Free Tier Size
    }
  }

  network_interface {
    network = "default"
    access_config {
      # Grants a public IP
    }
  }

  # 3. Startup Script (Installs Docker & n8n automatically)
  metadata_startup_script = <<-EOF
    #!/bin/bash
    apt-get update
    apt-get install -y docker.io
    systemctl start docker
    systemctl enable docker
    
    # Start n8n (The Orchestrator)
    docker run -d \
      --name n8n \
      -p 5678:5678 \
      -e GENERIC_TIMEZONE="America/New_York" \
      -v n8n_data:/home/node/.n8n \
      docker.n8n.io/n8nio/n8n
  EOF
}

# 4. Output the IP Address
output "external_ip" {
  value = google_compute_instance.sentinel_vm.network_interface[0].access_config[0].nat_ip
}