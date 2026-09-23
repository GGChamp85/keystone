# Copyright 2024-2026 Gaurav Gupta / Keystone — Apache 2.0

variable "aws_region" {
  description = "Region for the whole pilot. g6 (L4) instances are not in every region — check `aws ec2 describe-instance-type-offerings --location-type region --filters Name=instance-type,Values=g6.2xlarge`."
  type        = string
  default     = "us-east-1"
}

variable "cluster_name" {
  type    = string
  default = "keystone-pilot"
}

variable "kubernetes_version" {
  type    = string
  default = "1.33"
}

variable "system_instance_type" {
  type    = string
  default = "m6i.xlarge"
}

variable "system_node_count" {
  type    = number
  default = 2
}

variable "gpu_instance_type" {
  description = "See modules/aws-eks-gpu for why g6.2xlarge (one L4, 32 GiB host RAM) rather than g6.xlarge is the default."
  type        = string
  default     = "g6.2xlarge"
}

variable "gpu_node_count" {
  type    = number
  default = 1
}

variable "gpu_node_min" {
  type    = number
  default = 0
}

variable "gpu_node_max" {
  type    = number
  default = 2
}

variable "tags" {
  type = map(string)
  default = {
    Project     = "keystone"
    Environment = "pilot"
  }
}
