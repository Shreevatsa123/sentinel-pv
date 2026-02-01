# Sentinel-PV: Autonomous Pharmacovigilance Agent

### Configuration (Required)
To keep sensitive details private, this project uses a `terraform.tfvars` file which is excluded from version control. You must create this file manually.

1.  Navigate to `2_infrastructure/terraform/`.
2.  Create a file named `terraform.tfvars`.
3.  Add your specific Google Cloud Project ID inside:

```hcl
project_id = "your-gcp-project-id-here"
