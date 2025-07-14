# Infrastructure: 8.5/10 → 10/10 Upgrade Plan

## Current State (8.5/10)
✅ Cloud-native design with Modal deployment
✅ Container-ready with Docker/K8s configs
✅ Microservices patterns in architecture
✅ Proper secret management (after fixing hardcoded keys)
✅ Multi-environment support (dev/staging/prod)

## Missing for 10/10

### 1. Infrastructure as Code (IaC)

**File: `infrastructure/terraform/main.tf`**
```hcl
# Main Terraform configuration for mlTrainer infrastructure

terraform {
  required_version = ">= 1.0"
  
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.20"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.9"
    }
  }
  
  backend "s3" {
    bucket = "mltrainer-terraform-state"
    key    = "infrastructure/terraform.tfstate"
    region = "us-east-1"
    encrypt = true
    dynamodb_table = "mltrainer-terraform-locks"
  }
}

# Providers
provider "aws" {
  region = var.aws_region
}

provider "kubernetes" {
  host                   = module.eks.cluster_endpoint
  cluster_ca_certificate = base64decode(module.eks.cluster_certificate_authority_data)
  
  exec {
    api_version = "client.authentication.k8s.io/v1beta1"
    command     = "aws"
    args = ["eks", "get-token", "--cluster-name", module.eks.cluster_name]
  }
}

# Modules
module "networking" {
  source = "./modules/networking"
  
  vpc_cidr = var.vpc_cidr
  availability_zones = var.availability_zones
  public_subnet_cidrs = var.public_subnet_cidrs
  private_subnet_cidrs = var.private_subnet_cidrs
  
  tags = local.common_tags
}

module "eks" {
  source = "./modules/eks"
  
  cluster_name = "${var.project_name}-${var.environment}"
  cluster_version = var.kubernetes_version
  
  vpc_id = module.networking.vpc_id
  subnet_ids = module.networking.private_subnet_ids
  
  node_groups = {
    general = {
      desired_capacity = 3
      max_capacity     = 10
      min_capacity     = 3
      
      instance_types = ["t3.large"]
      
      k8s_labels = {
        Environment = var.environment
        Type        = "general"
      }
    }
    
    ml_gpu = {
      desired_capacity = 2
      max_capacity     = 5
      min_capacity     = 1
      
      instance_types = ["g4dn.xlarge"]
      
      k8s_labels = {
        Environment = var.environment
        Type        = "ml-gpu"
      }
      
      taints = [{
        key    = "nvidia.com/gpu"
        value  = "true"
        effect = "NO_SCHEDULE"
      }]
    }
  }
  
  tags = local.common_tags
}

module "rds" {
  source = "./modules/rds"
  
  identifier = "${var.project_name}-${var.environment}"
  
  engine         = "postgres"
  engine_version = "15.4"
  instance_class = var.db_instance_class
  
  allocated_storage = 100
  storage_encrypted = true
  
  vpc_id = module.networking.vpc_id
  subnet_ids = module.networking.private_subnet_ids
  
  backup_retention_period = 30
  backup_window          = "03:00-04:00"
  maintenance_window     = "sun:04:00-sun:05:00"
  
  tags = local.common_tags
}

module "redis" {
  source = "./modules/elasticache"
  
  cluster_id = "${var.project_name}-${var.environment}"
  
  node_type = var.redis_node_type
  num_cache_nodes = 3
  
  vpc_id = module.networking.vpc_id
  subnet_ids = module.networking.private_subnet_ids
  
  automatic_failover_enabled = true
  multi_az_enabled = true
  
  tags = local.common_tags
}

module "s3" {
  source = "./modules/s3"
  
  buckets = {
    models = {
      name = "${var.project_name}-models-${var.environment}"
      versioning = true
      lifecycle_rules = [{
        id = "archive-old-models"
        status = "Enabled"
        
        transition = [{
          days = 90
          storage_class = "GLACIER"
        }]
      }]
    }
    
    data = {
      name = "${var.project_name}-data-${var.environment}"
      versioning = true
      lifecycle_rules = [{
        id = "delete-old-data"
        status = "Enabled"
        
        expiration = {
          days = 365
        }
      }]
    }
  }
  
  tags = local.common_tags
}

module "monitoring" {
  source = "./modules/monitoring"
  
  cluster_name = module.eks.cluster_name
  
  prometheus_config = {
    retention_days = 30
    storage_size = "100Gi"
  }
  
  grafana_config = {
    admin_password = var.grafana_admin_password
  }
  
  alertmanager_config = {
    slack_webhook_url = var.slack_webhook_url
    pagerduty_key = var.pagerduty_key
  }
  
  tags = local.common_tags
}

# Outputs
output "eks_cluster_endpoint" {
  value = module.eks.cluster_endpoint
}

output "rds_endpoint" {
  value = module.rds.endpoint
}

output "redis_endpoint" {
  value = module.redis.endpoint
}

output "s3_buckets" {
  value = module.s3.bucket_arns
}
```

**File: `infrastructure/terraform/modules/eks/main.tf`**
```hcl
# EKS Cluster Module

resource "aws_eks_cluster" "main" {
  name     = var.cluster_name
  version  = var.cluster_version
  role_arn = aws_iam_role.cluster.arn
  
  vpc_config {
    subnet_ids              = var.subnet_ids
    endpoint_private_access = true
    endpoint_public_access  = true
    public_access_cidrs     = var.public_access_cidrs
  }
  
  encryption_config {
    provider {
      key_arn = aws_kms_key.eks.arn
    }
    resources = ["secrets"]
  }
  
  enabled_cluster_log_types = ["api", "audit", "authenticator", "controllerManager", "scheduler"]
  
  depends_on = [
    aws_iam_role_policy_attachment.cluster_AmazonEKSClusterPolicy,
    aws_iam_role_policy_attachment.cluster_AmazonEKSVPCResourceController,
  ]
}

# Node Groups
resource "aws_eks_node_group" "main" {
  for_each = var.node_groups
  
  cluster_name    = aws_eks_cluster.main.name
  node_group_name = each.key
  node_role_arn   = aws_iam_role.node.arn
  subnet_ids      = var.subnet_ids
  
  scaling_config {
    desired_size = each.value.desired_capacity
    max_size     = each.value.max_capacity
    min_size     = each.value.min_capacity
  }
  
  instance_types = each.value.instance_types
  
  labels = each.value.k8s_labels
  
  dynamic "taint" {
    for_each = lookup(each.value, "taints", [])
    content {
      key    = taint.value.key
      value  = taint.value.value
      effect = taint.value.effect
    }
  }
  
  depends_on = [
    aws_iam_role_policy_attachment.node_AmazonEKSWorkerNodePolicy,
    aws_iam_role_policy_attachment.node_AmazonEKS_CNI_Policy,
    aws_iam_role_policy_attachment.node_AmazonEC2ContainerRegistryReadOnly,
  ]
}

# IAM Roles
resource "aws_iam_role" "cluster" {
  name = "${var.cluster_name}-cluster-role"
  
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "eks.amazonaws.com"
      }
    }]
  })
}

resource "aws_iam_role" "node" {
  name = "${var.cluster_name}-node-role"
  
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "ec2.amazonaws.com"
      }
    }]
  })
}

# KMS Key for encryption
resource "aws_kms_key" "eks" {
  description = "EKS cluster encryption key"
  
  tags = var.tags
}

# Add-ons
resource "aws_eks_addon" "addons" {
  for_each = toset([
    "kube-proxy",
    "vpc-cni",
    "coredns",
    "aws-ebs-csi-driver"
  ])
  
  cluster_name = aws_eks_cluster.main.name
  addon_name   = each.value
}
```

### 2. GitOps Workflow

**File: `gitops/argocd/applications/mltrainer-app.yaml`**
```yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: mltrainer
  namespace: argocd
  finalizers:
    - resources-finalizer.argocd.argoproj.io
spec:
  project: default
  
  source:
    repoURL: https://github.com/yourusername/mltrainer
    targetRevision: HEAD
    path: k8s/overlays/production
    
    # Kustomize
    kustomize:
      images:
      - mltrainer/api:v1.2.3
      - mltrainer/ui:v1.2.3
      
  destination:
    server: https://kubernetes.default.svc
    namespace: mltrainer
    
  syncPolicy:
    automated:
      prune: true
      selfHeal: true
      allowEmpty: false
    syncOptions:
    - CreateNamespace=true
    - PrunePropagationPolicy=foreground
    - PruneLast=true
    
    retry:
      limit: 5
      backoff:
        duration: 5s
        factor: 2
        maxDuration: 3m
        
  revisionHistoryLimit: 10
  
  # Health checks
  health:
    enabled: true

---
apiVersion: argoproj.io/v1alpha1
kind: AppProject
metadata:
  name: mltrainer-project
  namespace: argocd
spec:
  description: mlTrainer ML Platform
  
  sourceRepos:
  - 'https://github.com/yourusername/*'
  
  destinations:
  - namespace: 'mltrainer'
    server: https://kubernetes.default.svc
  - namespace: 'mltrainer-staging'
    server: https://kubernetes.default.svc
    
  clusterResourceWhitelist:
  - group: ''
    kind: Namespace
  - group: rbac.authorization.k8s.io
    kind: ClusterRole
  - group: rbac.authorization.k8s.io
    kind: ClusterRoleBinding
    
  namespaceResourceWhitelist:
  - group: '*'
    kind: '*'
    
  roles:
  - name: admin
    policies:
    - p, proj:mltrainer-project:admin, applications, *, mltrainer-project/*, allow
    groups:
    - mltrainer:admins
    
  - name: readonly
    policies:
    - p, proj:mltrainer-project:readonly, applications, get, mltrainer-project/*, allow
    groups:
    - mltrainer:developers
```

**File: `gitops/flux/clusters/production/mltrainer-kustomization.yaml`**
```yaml
apiVersion: kustomize.toolkit.fluxcd.io/v1
kind: Kustomization
metadata:
  name: mltrainer
  namespace: flux-system
spec:
  interval: 10m
  path: "./k8s/overlays/production"
  prune: true
  sourceRef:
    kind: GitRepository
    name: mltrainer
  targetNamespace: mltrainer
  
  # Health checks
  healthChecks:
    - apiVersion: apps/v1
      kind: Deployment
      name: mltrainer-api
      namespace: mltrainer
    - apiVersion: apps/v1
      kind: Deployment
      name: mltrainer-ui
      namespace: mltrainer
      
  # Dependencies
  dependsOn:
    - name: infrastructure
    - name: cert-manager
    
  # Post build substitutions
  postBuild:
    substitute:
      cluster_name: "production"
      region: "us-east-1"
    substituteFrom:
      - kind: ConfigMap
        name: cluster-config
        
  # Validation
  validation: client
  
  # Decryption
  decryption:
    provider: sops
    secretRef:
      name: sops-gpg

---
apiVersion: source.toolkit.fluxcd.io/v1
kind: GitRepository
metadata:
  name: mltrainer
  namespace: flux-system
spec:
  interval: 1m
  ref:
    branch: main
  secretRef:
    name: mltrainer-repo
  url: ssh://git@github.com/yourusername/mltrainer
  
---
apiVersion: notification.toolkit.fluxcd.io/v1beta1
kind: Alert
metadata:
  name: mltrainer-alerts
  namespace: flux-system
spec:
  providerRef:
    name: slack
  eventSeverity: info
  eventSources:
    - kind: GitRepository
      name: mltrainer
    - kind: Kustomization
      name: mltrainer
  summary: 'mlTrainer deployment notifications'
```

### 3. Service Mesh (Istio)

**File: `k8s/istio/virtual-service.yaml`**
```yaml
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: mltrainer-api
  namespace: mltrainer
spec:
  hosts:
  - api.mltrainer.com
  gateways:
  - mltrainer-gateway
  http:
  - match:
    - uri:
        prefix: /v1/
    route:
    - destination:
        host: mltrainer-api
        port:
          number: 8000
      weight: 100
    timeout: 30s
    retries:
      attempts: 3
      perTryTimeout: 10s
      retryOn: 5xx,reset,connect-failure,refused-stream
      
  - match:
    - uri:
        prefix: /v2/
    route:
    - destination:
        host: mltrainer-api-v2
        port:
          number: 8000
      weight: 10  # Canary deployment
    - destination:
        host: mltrainer-api
        port:
          number: 8000
      weight: 90
      
---
apiVersion: networking.istio.io/v1beta1
kind: DestinationRule
metadata:
  name: mltrainer-api
  namespace: mltrainer
spec:
  host: mltrainer-api
  trafficPolicy:
    connectionPool:
      tcp:
        maxConnections: 100
      http:
        http1MaxPendingRequests: 100
        http2MaxRequests: 100
        maxRequestsPerConnection: 2
    loadBalancer:
      simple: LEAST_REQUEST
    outlierDetection:
      consecutiveErrors: 5
      interval: 30s
      baseEjectionTime: 30s
      maxEjectionPercent: 50
      minHealthPercent: 50
      splitExternalLocalOriginErrors: true
      
---
apiVersion: security.istio.io/v1beta1
kind: PeerAuthentication
metadata:
  name: default
  namespace: mltrainer
spec:
  mtls:
    mode: STRICT
    
---
apiVersion: security.istio.io/v1beta1
kind: AuthorizationPolicy
metadata:
  name: mltrainer-api
  namespace: mltrainer
spec:
  selector:
    matchLabels:
      app: mltrainer-api
  action: ALLOW
  rules:
  - from:
    - source:
        principals: ["cluster.local/ns/mltrainer/sa/mltrainer-ui"]
    to:
    - operation:
        methods: ["GET", "POST"]
        paths: ["/api/*"]
        
---
apiVersion: networking.istio.io/v1beta1
kind: ServiceEntry
metadata:
  name: external-polygon-api
  namespace: mltrainer
spec:
  hosts:
  - api.polygon.io
  ports:
  - number: 443
    name: https
    protocol: HTTPS
  location: MESH_EXTERNAL
  resolution: DNS
  
---
apiVersion: networking.istio.io/v1beta1
kind: Gateway
metadata:
  name: mltrainer-gateway
  namespace: mltrainer
spec:
  selector:
    istio: ingressgateway
  servers:
  - port:
      number: 443
      name: https
      protocol: HTTPS
    tls:
      mode: SIMPLE
      credentialName: mltrainer-tls
    hosts:
    - "*.mltrainer.com"
```

### 4. Multi-Region Deployment

**File: `infrastructure/terraform/multi-region/main.tf`**
```hcl
# Multi-region deployment configuration

locals {
  regions = {
    primary = {
      region = "us-east-1"
      vpc_cidr = "10.0.0.0/16"
      availability_zones = ["us-east-1a", "us-east-1b", "us-east-1c"]
    }
    secondary = {
      region = "eu-west-1"
      vpc_cidr = "10.1.0.0/16"
      availability_zones = ["eu-west-1a", "eu-west-1b", "eu-west-1c"]
    }
    dr = {
      region = "ap-southeast-1"
      vpc_cidr = "10.2.0.0/16"
      availability_zones = ["ap-southeast-1a", "ap-southeast-1b", "ap-southeast-1c"]
    }
  }
}

# Deploy infrastructure in each region
module "regional_infrastructure" {
  for_each = local.regions
  source = "../modules/regional"
  
  providers = {
    aws = aws[each.key]
  }
  
  region_name = each.key
  region = each.value.region
  vpc_cidr = each.value.vpc_cidr
  availability_zones = each.value.availability_zones
  
  # Cross-region replication
  enable_cross_region_replication = true
  replication_regions = [for k, v in local.regions : v.region if k != each.key]
  
  tags = merge(local.common_tags, {
    Region = each.key
    MultiRegion = "true"
  })
}

# Global Route53 configuration
resource "aws_route53_zone" "main" {
  name = "mltrainer.com"
  
  tags = local.common_tags
}

# Health checks for each region
resource "aws_route53_health_check" "regional" {
  for_each = local.regions
  
  fqdn              = "${each.key}.mltrainer.com"
  port              = 443
  type              = "HTTPS"
  resource_path     = "/health"
  failure_threshold = "3"
  request_interval  = "30"
  
  tags = merge(local.common_tags, {
    Region = each.key
  })
}

# Geolocation routing
resource "aws_route53_record" "geolocation" {
  for_each = {
    us = {
      region = "primary"
      continent = "NA"
    }
    eu = {
      region = "secondary"
      continent = "EU"
    }
    asia = {
      region = "dr"
      continent = "AS"
    }
  }
  
  zone_id = aws_route53_zone.main.zone_id
  name    = "api.mltrainer.com"
  type    = "A"
  
  alias {
    name                   = module.regional_infrastructure[each.value.region].load_balancer_dns
    zone_id                = module.regional_infrastructure[each.value.region].load_balancer_zone_id
    evaluate_target_health = true
  }
  
  set_identifier = each.key
  
  geolocation_routing_policy {
    continent = each.value.continent
  }
}

# Failover routing
resource "aws_route53_record" "failover_primary" {
  zone_id = aws_route53_zone.main.zone_id
  name    = "api.mltrainer.com"
  type    = "A"
  
  alias {
    name                   = module.regional_infrastructure["primary"].load_balancer_dns
    zone_id                = module.regional_infrastructure["primary"].load_balancer_zone_id
    evaluate_target_health = true
  }
  
  set_identifier = "primary"
  
  failover_routing_policy {
    type = "PRIMARY"
  }
  
  health_check_id = aws_route53_health_check.regional["primary"].id
}

resource "aws_route53_record" "failover_secondary" {
  zone_id = aws_route53_zone.main.zone_id
  name    = "api.mltrainer.com"
  type    = "A"
  
  alias {
    name                   = module.regional_infrastructure["secondary"].load_balancer_dns
    zone_id                = module.regional_infrastructure["secondary"].load_balancer_zone_id
    evaluate_target_health = true
  }
  
  set_identifier = "secondary"
  
  failover_routing_policy {
    type = "SECONDARY"
  }
}

# Global database with read replicas
module "global_database" {
  source = "../modules/aurora-global"
  
  global_cluster_identifier = "${var.project_name}-global"
  
  primary_cluster = {
    region = local.regions.primary.region
    cluster_identifier = "${var.project_name}-primary"
    vpc_id = module.regional_infrastructure["primary"].vpc_id
    subnet_ids = module.regional_infrastructure["primary"].database_subnet_ids
  }
  
  secondary_clusters = {
    secondary = {
      region = local.regions.secondary.region
      cluster_identifier = "${var.project_name}-secondary"
      vpc_id = module.regional_infrastructure["secondary"].vpc_id
      subnet_ids = module.regional_infrastructure["secondary"].database_subnet_ids
    }
    dr = {
      region = local.regions.dr.region
      cluster_identifier = "${var.project_name}-dr"
      vpc_id = module.regional_infrastructure["dr"].vpc_id
      subnet_ids = module.regional_infrastructure["dr"].database_subnet_ids
    }
  }
  
  engine = "aurora-postgresql"
  engine_version = "15.4"
  
  tags = local.common_tags
}
```

### 5. Disaster Recovery Plan

**File: `docs/DISASTER_RECOVERY_PLAN.md`**
```markdown
# mlTrainer Disaster Recovery Plan

## Overview
This document outlines the disaster recovery (DR) procedures for the mlTrainer platform.

## Recovery Objectives
- **RTO (Recovery Time Objective)**: 4 hours
- **RPO (Recovery Point Objective)**: 1 hour

## Infrastructure Overview

### Primary Region: US-East-1
- EKS Cluster: mltrainer-production
- RDS: Aurora PostgreSQL (Global Database)
- ElastiCache: Redis Cluster
- S3: Model storage (cross-region replication)

### Secondary Region: EU-West-1
- EKS Cluster: mltrainer-secondary
- RDS: Aurora Read Replica (promotable)
- ElastiCache: Redis Cluster
- S3: Replicated buckets

### DR Region: AP-Southeast-1
- EKS Cluster: mltrainer-dr (minimal capacity)
- RDS: Aurora Read Replica (promotable)
- S3: Replicated buckets

## Disaster Scenarios

### 1. Service Failure
**Detection**: Health checks failing, alerts triggered
**Response**:
1. Check service logs: `kubectl logs -n mltrainer deployment/mltrainer-api`
2. Restart pods: `kubectl rollout restart -n mltrainer deployment/mltrainer-api`
3. If persistent, rollback: `kubectl rollout undo -n mltrainer deployment/mltrainer-api`

### 2. Database Failure
**Detection**: Connection errors, Aurora alarms
**Response**:
1. Check Aurora cluster status in AWS Console
2. If primary failed, promote secondary:
   ```bash
   aws rds promote-read-replica-db-cluster \
     --db-cluster-identifier mltrainer-secondary
   ```
3. Update application connection strings
4. Verify data consistency

### 3. Region Failure
**Detection**: Multiple service failures, AWS Health Dashboard
**Response**:
1. Execute region failover runbook:
   ```bash
   ./scripts/regional-failover.sh --from us-east-1 --to eu-west-1
   ```
2. Update Route53 to point to secondary region
3. Scale up secondary region capacity
4. Notify stakeholders

## Backup Procedures

### Database Backups
- **Automated**: Daily snapshots, 30-day retention
- **Manual**: Before major deployments
- **Location**: S3 with cross-region replication

### Model Backups
- **Frequency**: After each training session
- **Location**: S3 with versioning and lifecycle policies
- **Restore**: Use model registry to restore specific versions

### Configuration Backups
- **Method**: GitOps - all configs in Git
- **Secrets**: Encrypted with SOPS, backed up in AWS Secrets Manager

## Recovery Procedures

### 1. Data Recovery
```bash
# Restore database from snapshot
aws rds restore-db-cluster-from-snapshot \
  --db-cluster-identifier mltrainer-restored \
  --snapshot-identifier mltrainer-snapshot-20240710

# Restore models from S3
aws s3 sync s3://mltrainer-models-backup/20240710/ ./models/
```

### 2. Service Recovery
```bash
# Apply Kubernetes manifests
kubectl apply -k k8s/overlays/disaster-recovery/

# Verify deployments
kubectl get deployments -n mltrainer

# Run smoke tests
./scripts/smoke-tests.sh
```

### 3. Regional Failover
```bash
# Promote DR region
terraform workspace select dr
terraform apply -var="enable_dr_mode=true"

# Update DNS
aws route53 change-resource-record-sets \
  --hosted-zone-id Z1234567890ABC \
  --change-batch file://dr-dns-change.json
```

## Testing Procedures

### Monthly DR Drills
1. **Backup Restoration Test**
   - Restore database to test environment
   - Verify data integrity
   - Test model loading

2. **Failover Test**
   - Simulate region failure in staging
   - Execute failover procedures
   - Measure RTO/RPO

3. **Chaos Engineering**
   - Run chaos experiments
   - Verify auto-recovery
   - Update runbooks based on findings

## Communication Plan

### Incident Levels
- **P1**: Complete service outage
- **P2**: Partial service degradation
- **P3**: Non-critical component failure

### Notification Channels
1. **PagerDuty**: Immediate alerts for on-call
2. **Slack**: #incidents channel for coordination
3. **Email**: Stakeholder updates
4. **Status Page**: Public communication

### Escalation Path
1. On-call engineer (0-15 min)
2. Team lead (15-30 min)
3. Engineering manager (30-60 min)
4. CTO (60+ min)

## Post-Incident Procedures

1. **Incident Report**
   - Timeline of events
   - Root cause analysis
   - Actions taken
   - Lessons learned

2. **Runbook Updates**
   - Update procedures based on incident
   - Add new scenarios discovered
   - Improve automation

3. **Testing Updates**
   - Add regression tests
   - Update chaos experiments
   - Enhance monitoring

## Automation Scripts

### Regional Failover Script
```bash
#!/bin/bash
# scripts/regional-failover.sh

set -e

FROM_REGION=$1
TO_REGION=$2

echo "Starting failover from $FROM_REGION to $TO_REGION"

# 1. Update Kubernetes context
kubectl config use-context $TO_REGION

# 2. Scale up capacity
kubectl scale deployment -n mltrainer --all --replicas=5

# 3. Promote database
aws rds promote-read-replica-db-cluster \
  --db-cluster-identifier mltrainer-$TO_REGION \
  --region $TO_REGION

# 4. Update Route53
aws route53 change-resource-record-sets \
  --hosted-zone-id $HOSTED_ZONE_ID \
  --change-batch file://failover-dns-$TO_REGION.json

# 5. Verify health
./scripts/health-check.sh --region $TO_REGION

echo "Failover completed"
```

## Recovery Validation

### Checklist
- [ ] All services responding to health checks
- [ ] Database connections established
- [ ] Model serving operational
- [ ] External APIs accessible
- [ ] Monitoring and alerting functional
- [ ] Logs being collected
- [ ] Backups resuming
- [ ] Performance within SLA

### Success Metrics
- Service availability > 99.9%
- API latency < 200ms p99
- Zero data loss
- All models accessible
- All integrations functional
```

## Implementation Priority

1. **Week 1**: Terraform infrastructure setup
2. **Week 2**: GitOps implementation
3. **Week 3**: Service mesh deployment
4. **Week 4**: Multi-region configuration
5. **Week 5**: DR testing and documentation

## Success Metrics

- 100% infrastructure defined as code
- Automated GitOps deployments
- Service mesh providing mTLS and traffic management
- Multi-region active-active deployment
- DR drills achieving <4 hour RTO