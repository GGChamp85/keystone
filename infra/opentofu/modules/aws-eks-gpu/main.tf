# Copyright 2024-2026 Gaurav Gupta / Keystone — Apache 2.0
#
# One EKS cluster sized for a Keystone GPU pilot: a private-subnet VPC, a
# CPU system node group for the stateful stack, one NVIDIA-L4 GPU node
# group for the vLLM coding role (tainted so only GPU workloads land on
# it), the EBS and EFS CSI drivers wired through EKS Pod Identity, an EFS
# file system exposed as an RWX StorageClass for the shared model cache, a
# gp3 StorageClass for Postgres/Redis/Qdrant/OpenBao, and NVIDIA's
# k8s-device-plugin so `nvidia.com/gpu` is schedulable.
#
# Drivers: the EKS-optimized AL2023 NVIDIA AMI ships the NVIDIA kernel
# driver and container toolkit; it does NOT ship the device plugin, which
# is why this module installs the nvidia-device-plugin Helm chart itself.
#
# Verified so far: `tofu fmt -check`, `tofu init -backend=false`,
# `tofu validate` (CI job iac-validate). No real `apply` has been run on an
# AWS account yet — see docs/deployment/verification-log.md.
#
# The kubernetes and helm providers are configured by the calling root
# module from this module's cluster outputs (see
# environments/aws-pilot/main.tf) — the usual EKS pattern, because a
# provider cannot be configured from inside the module that creates the
# cluster it points at.

data "aws_availability_zones" "available" {
  state = "available"

  filter {
    name   = "opt-in-status"
    values = ["opt-in-not-required"]
  }
}

data "aws_region" "current" {}

locals {
  azs = slice(data.aws_availability_zones.available.names, 0, 2)
  tags = merge(var.tags, {
    "keystone.io/cluster" = var.cluster_name
    "ManagedBy"           = "opentofu"
  })
  gpu_taint_key    = "nvidia.com/gpu"
  gpu_taint_value  = "present"
  gpu_taint_effect = "NoSchedule"
}

# ---------------------------------------------------------------------------
# Network: one VPC, two AZs, public subnets for the NAT gateway / load
# balancers, private subnets for every node.
# ---------------------------------------------------------------------------

resource "aws_vpc" "this" {
  cidr_block           = var.vpc_cidr
  enable_dns_support   = true
  enable_dns_hostnames = true

  tags = merge(local.tags, { Name = "${var.cluster_name}-vpc" })
}

resource "aws_subnet" "public" {
  count = 2

  vpc_id                  = aws_vpc.this.id
  cidr_block              = cidrsubnet(var.vpc_cidr, 4, count.index)
  availability_zone       = local.azs[count.index]
  map_public_ip_on_launch = true

  tags = merge(local.tags, {
    Name                                        = "${var.cluster_name}-public-${local.azs[count.index]}"
    "kubernetes.io/role/elb"                    = "1"
    "kubernetes.io/cluster/${var.cluster_name}" = "shared"
  })
}

resource "aws_subnet" "private" {
  count = 2

  vpc_id            = aws_vpc.this.id
  cidr_block        = cidrsubnet(var.vpc_cidr, 4, count.index + 2)
  availability_zone = local.azs[count.index]

  tags = merge(local.tags, {
    Name                                        = "${var.cluster_name}-private-${local.azs[count.index]}"
    "kubernetes.io/role/internal-elb"           = "1"
    "kubernetes.io/cluster/${var.cluster_name}" = "shared"
  })
}

resource "aws_internet_gateway" "this" {
  vpc_id = aws_vpc.this.id
  tags   = merge(local.tags, { Name = "${var.cluster_name}-igw" })
}

resource "aws_eip" "nat" {
  domain = "vpc"
  tags   = merge(local.tags, { Name = "${var.cluster_name}-nat" })
}

# One NAT gateway (not one per AZ): a pilot trades AZ-level NAT redundancy
# for roughly half the NAT hourly cost. Node egress (image pulls, weight
# downloads) goes through it; inbound traffic does not.
resource "aws_nat_gateway" "this" {
  allocation_id = aws_eip.nat.id
  subnet_id     = aws_subnet.public[0].id
  tags          = merge(local.tags, { Name = "${var.cluster_name}-nat" })

  depends_on = [aws_internet_gateway.this]
}

resource "aws_route_table" "public" {
  vpc_id = aws_vpc.this.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.this.id
  }

  tags = merge(local.tags, { Name = "${var.cluster_name}-public" })
}

resource "aws_route_table" "private" {
  vpc_id = aws_vpc.this.id

  route {
    cidr_block     = "0.0.0.0/0"
    nat_gateway_id = aws_nat_gateway.this.id
  }

  tags = merge(local.tags, { Name = "${var.cluster_name}-private" })
}

resource "aws_route_table_association" "public" {
  count = 2

  subnet_id      = aws_subnet.public[count.index].id
  route_table_id = aws_route_table.public.id
}

resource "aws_route_table_association" "private" {
  count = 2

  subnet_id      = aws_subnet.private[count.index].id
  route_table_id = aws_route_table.private.id
}

# ---------------------------------------------------------------------------
# IAM: control plane role, node role, and Pod Identity roles for the two
# CSI drivers (no OIDC provider / IRSA thumbprint dance needed).
# ---------------------------------------------------------------------------

data "aws_iam_policy_document" "eks_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["eks.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "cluster" {
  name               = "${var.cluster_name}-cluster"
  assume_role_policy = data.aws_iam_policy_document.eks_assume.json
  tags               = local.tags
}

resource "aws_iam_role_policy_attachment" "cluster" {
  role       = aws_iam_role.cluster.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSClusterPolicy"
}

data "aws_iam_policy_document" "ec2_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ec2.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "node" {
  name               = "${var.cluster_name}-node"
  assume_role_policy = data.aws_iam_policy_document.ec2_assume.json
  tags               = local.tags
}

resource "aws_iam_role_policy_attachment" "node" {
  for_each = toset([
    "arn:aws:iam::aws:policy/AmazonEKSWorkerNodePolicy",
    "arn:aws:iam::aws:policy/AmazonEKS_CNI_Policy",
    "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly",
  ])

  role       = aws_iam_role.node.name
  policy_arn = each.value
}

data "aws_iam_policy_document" "pod_identity_assume" {
  statement {
    actions = ["sts:AssumeRole", "sts:TagSession"]
    principals {
      type        = "Service"
      identifiers = ["pods.eks.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "ebs_csi" {
  name               = "${var.cluster_name}-ebs-csi"
  assume_role_policy = data.aws_iam_policy_document.pod_identity_assume.json
  tags               = local.tags
}

resource "aws_iam_role_policy_attachment" "ebs_csi" {
  role       = aws_iam_role.ebs_csi.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonEBSCSIDriverPolicy"
}

resource "aws_iam_role" "efs_csi" {
  name               = "${var.cluster_name}-efs-csi"
  assume_role_policy = data.aws_iam_policy_document.pod_identity_assume.json
  tags               = local.tags
}

resource "aws_iam_role_policy_attachment" "efs_csi" {
  role       = aws_iam_role.efs_csi.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonEFSCSIDriverPolicy"
}

# ---------------------------------------------------------------------------
# Control plane
# ---------------------------------------------------------------------------

resource "aws_eks_cluster" "this" {
  name     = var.cluster_name
  role_arn = aws_iam_role.cluster.arn
  version  = var.kubernetes_version

  vpc_config {
    subnet_ids              = concat(aws_subnet.private[*].id, aws_subnet.public[*].id)
    endpoint_public_access  = true
    endpoint_private_access = true
  }

  access_config {
    authentication_mode                         = "API_AND_CONFIG_MAP"
    bootstrap_cluster_creator_admin_permissions = true
  }

  tags = local.tags

  depends_on = [aws_iam_role_policy_attachment.cluster]
}

# ---------------------------------------------------------------------------
# Node groups
# ---------------------------------------------------------------------------

resource "aws_eks_node_group" "system" {
  cluster_name    = aws_eks_cluster.this.name
  node_group_name = "${var.cluster_name}-system"
  node_role_arn   = aws_iam_role.node.arn
  subnet_ids      = aws_subnet.private[*].id
  instance_types  = [var.system_instance_type]
  ami_type        = "AL2023_x86_64_STANDARD"
  capacity_type   = "ON_DEMAND"
  disk_size       = 100

  scaling_config {
    desired_size = var.system_node_count
    min_size     = var.system_node_min
    max_size     = var.system_node_max
  }

  update_config {
    max_unavailable = 1
  }

  labels = {
    nodepool = "system"
  }

  tags = local.tags

  depends_on = [aws_iam_role_policy_attachment.node]

  lifecycle {
    ignore_changes = [scaling_config[0].desired_size] # a cluster autoscaler, if added, owns this
  }
}

resource "aws_eks_node_group" "gpu" {
  cluster_name    = aws_eks_cluster.this.name
  node_group_name = "${var.cluster_name}-gpu"
  node_role_arn   = aws_iam_role.node.arn
  subnet_ids      = aws_subnet.private[*].id
  instance_types  = [var.gpu_instance_type]
  ami_type        = var.gpu_ami_type
  capacity_type   = "ON_DEMAND"
  disk_size       = var.gpu_node_disk_gb

  scaling_config {
    desired_size = var.gpu_node_count
    min_size     = var.gpu_node_min
    max_size     = var.gpu_node_max
  }

  update_config {
    max_unavailable = 1
  }

  labels = {
    nodepool                 = var.gpu_nodepool_label
    "nvidia.com/gpu.present" = "true"
  }

  # Keeps every non-GPU pod off the expensive nodes. The chart's vLLM and
  # training pods carry the matching toleration (templates/vllm.yaml,
  # templates/training-runtime.yaml), as does the device plugin below.
  taint {
    key    = local.gpu_taint_key
    value  = local.gpu_taint_value
    effect = "NO_SCHEDULE"
  }

  tags = local.tags

  depends_on = [aws_iam_role_policy_attachment.node]

  lifecycle {
    ignore_changes = [scaling_config[0].desired_size]
  }
}

# ---------------------------------------------------------------------------
# Managed add-ons. Pod Identity replaces IRSA for the CSI drivers: the
# agent add-on must exist before an association can bind.
# ---------------------------------------------------------------------------

resource "aws_eks_addon" "pod_identity_agent" {
  cluster_name                = aws_eks_cluster.this.name
  addon_name                  = "eks-pod-identity-agent"
  resolve_conflicts_on_create = "OVERWRITE"
  resolve_conflicts_on_update = "OVERWRITE"
  tags                        = local.tags

  depends_on = [aws_eks_node_group.system]
}

resource "aws_eks_addon" "vpc_cni" {
  cluster_name                = aws_eks_cluster.this.name
  addon_name                  = "vpc-cni"
  resolve_conflicts_on_create = "OVERWRITE"
  resolve_conflicts_on_update = "OVERWRITE"
  tags                        = local.tags
}

resource "aws_eks_addon" "kube_proxy" {
  cluster_name                = aws_eks_cluster.this.name
  addon_name                  = "kube-proxy"
  resolve_conflicts_on_create = "OVERWRITE"
  resolve_conflicts_on_update = "OVERWRITE"
  tags                        = local.tags
}

resource "aws_eks_addon" "coredns" {
  cluster_name                = aws_eks_cluster.this.name
  addon_name                  = "coredns"
  resolve_conflicts_on_create = "OVERWRITE"
  resolve_conflicts_on_update = "OVERWRITE"
  tags                        = local.tags

  depends_on = [aws_eks_node_group.system] # CoreDNS pods need somewhere to run before the add-on reports ACTIVE
}

resource "aws_eks_addon" "ebs_csi" {
  cluster_name                = aws_eks_cluster.this.name
  addon_name                  = "aws-ebs-csi-driver"
  resolve_conflicts_on_create = "OVERWRITE"
  resolve_conflicts_on_update = "OVERWRITE"
  tags                        = local.tags

  pod_identity_association {
    role_arn        = aws_iam_role.ebs_csi.arn
    service_account = "ebs-csi-controller-sa"
  }

  depends_on = [aws_eks_addon.pod_identity_agent, aws_iam_role_policy_attachment.ebs_csi]
}

resource "aws_eks_addon" "efs_csi" {
  cluster_name                = aws_eks_cluster.this.name
  addon_name                  = "aws-efs-csi-driver"
  resolve_conflicts_on_create = "OVERWRITE"
  resolve_conflicts_on_update = "OVERWRITE"
  tags                        = local.tags

  pod_identity_association {
    role_arn        = aws_iam_role.efs_csi.arn
    service_account = "efs-csi-controller-sa"
  }

  depends_on = [aws_eks_addon.pod_identity_agent, aws_iam_role_policy_attachment.efs_csi]
}

# ---------------------------------------------------------------------------
# EFS: the RWX model cache shared by every GPU node.
# ---------------------------------------------------------------------------

resource "aws_security_group" "efs" {
  name        = "${var.cluster_name}-efs"
  description = "NFS from the cluster VPC to the Keystone model-cache EFS"
  vpc_id      = aws_vpc.this.id

  ingress {
    description = "NFS from nodes"
    from_port   = 2049
    to_port     = 2049
    protocol    = "tcp"
    cidr_blocks = [var.vpc_cidr]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = merge(local.tags, { Name = "${var.cluster_name}-efs" })
}

resource "aws_efs_file_system" "model_cache" {
  creation_token   = "${var.cluster_name}-model-cache"
  encrypted        = true
  performance_mode = "generalPurpose"
  throughput_mode  = var.efs_throughput_mode

  tags = merge(local.tags, { Name = "${var.cluster_name}-model-cache" })
}

resource "aws_efs_mount_target" "model_cache" {
  count = 2

  file_system_id  = aws_efs_file_system.model_cache.id
  subnet_id       = aws_subnet.private[count.index].id
  security_groups = [aws_security_group.efs.id]
}

# ---------------------------------------------------------------------------
# In-cluster: StorageClasses and the NVIDIA device plugin.
# ---------------------------------------------------------------------------

resource "kubernetes_storage_class_v1" "efs_rwx" {
  metadata {
    name = "keystone-efs-rwx"
  }

  storage_provisioner = "efs.csi.aws.com"
  reclaim_policy      = "Delete" # a pilot's model cache is re-downloadable; switch to Retain for production weights
  volume_binding_mode = "Immediate"

  parameters = {
    provisioningMode = "efs-ap" # one EFS access point per PVC
    fileSystemId     = aws_efs_file_system.model_cache.id
    directoryPerms   = "700"
    basePath         = "/keystone"
  }

  mount_options = ["tls"]

  depends_on = [aws_eks_addon.efs_csi, aws_efs_mount_target.model_cache]
}

resource "kubernetes_storage_class_v1" "gp3" {
  metadata {
    name = "keystone-gp3"
  }

  storage_provisioner    = "ebs.csi.aws.com"
  reclaim_policy         = "Delete"
  volume_binding_mode    = "WaitForFirstConsumer" # bind in the AZ the pod lands in
  allow_volume_expansion = true

  parameters = {
    type      = "gp3"
    encrypted = "true"
    fsType    = "ext4"
  }

  depends_on = [aws_eks_addon.ebs_csi]
}

resource "helm_release" "nvidia_device_plugin" {
  count = var.install_nvidia_device_plugin ? 1 : 0

  name             = "nvidia-device-plugin"
  repository       = "https://nvidia.github.io/k8s-device-plugin"
  chart            = "nvidia-device-plugin"
  version          = var.nvidia_device_plugin_version
  namespace        = "nvidia-device-plugin"
  create_namespace = true
  wait             = true
  timeout          = 600

  # Only run the plugin DaemonSet on GPU nodes; the chart's default
  # tolerations already cover the nvidia.com/gpu NoSchedule taint.
  set {
    name  = "nodeSelector.nodepool"
    value = var.gpu_nodepool_label
  }

  depends_on = [aws_eks_node_group.gpu, aws_eks_addon.coredns]
}
