# 🌐 Custom Domain Setup for mlTrainer on windfuhr.net

This guide will help you configure windfuhr.net to work with your Modal deployment.

## 📋 Prerequisites

- [ ] Access to windfuhr.net DNS settings (registrar/hosting panel)
- [ ] Modal account with deployed mlTrainer app
- [ ] Your Modal workspace name (find it in Modal dashboard)

## 🔧 Step 1: Get Your Modal Endpoints

First, deploy to Modal and note your endpoints:

```bash
# Deploy the app
modal deploy modal_app.py

# Your endpoints will be:
# https://[WORKSPACE]--mltrainer-app.modal.run
# https://[WORKSPACE]--mltrainer-api.modal.run
# https://[WORKSPACE]--mltrainer-monitor.modal.run
```

Replace `[WORKSPACE]` with your actual Modal workspace name.

## 🌐 Step 2: Configure DNS Records

Add these DNS records to windfuhr.net:

### Option A: Using CNAME Records (Recommended)

| Type | Name | Value | TTL |
|------|------|-------|-----|
| CNAME | mltrainer | `[WORKSPACE]--mltrainer-app.modal.run` | 3600 |
| CNAME | api.mltrainer | `[WORKSPACE]--mltrainer-api.modal.run` | 3600 |
| CNAME | monitor.mltrainer | `[WORKSPACE]--mltrainer-monitor.modal.run` | 3600 |

### Option B: Using A Records + Modal's IP

1. Get Modal's IP:
```bash
nslookup [WORKSPACE]--mltrainer-app.modal.run
```

2. Add A records:

| Type | Name | Value | TTL |
|------|------|-------|-----|
| A | mltrainer | `[MODAL_IP]` | 3600 |
| A | api.mltrainer | `[MODAL_IP]` | 3600 |
| A | monitor.mltrainer | `[MODAL_IP]` | 3600 |

## 🔒 Step 3: Configure Custom Domains in Modal

1. Go to Modal Dashboard → Settings → Custom Domains
2. Add your domains:
   - `mltrainer.windfuhr.net`
   - `api.mltrainer.windfuhr.net`
   - `monitor.mltrainer.windfuhr.net`

3. Modal will provide verification records. Add these to your DNS:

| Type | Name | Value | TTL |
|------|------|-------|-----|
| TXT | _modal-verification.mltrainer | `modal-verify-xxxxx` | 3600 |

## 🛡️ Step 4: SSL Certificate Setup

Modal automatically provides SSL certificates via Let's Encrypt. Once DNS is configured:

1. Modal will automatically request certificates
2. Wait 5-10 minutes for propagation
3. Check status in Modal dashboard

## 🔗 Step 5: Update Your Application

Update your Modal app to recognize custom domains:

```python
# In modal_app.py, add:
CUSTOM_DOMAINS = {
    "mltrainer-app": "mltrainer.windfuhr.net",
    "mltrainer-api": "api.mltrainer.windfuhr.net",
    "mltrainer-monitor": "monitor.mltrainer.windfuhr.net"
}

# Update endpoints to return custom domains
@stub.function()
@modal.web_endpoint(label="mltrainer-info")
def info():
    return {
        "endpoints": {
            "app": "https://mltrainer.windfuhr.net",
            "api": "https://api.mltrainer.windfuhr.net",
            "monitor": "https://monitor.mltrainer.windfuhr.net"
        }
    }
```

## ✅ Step 6: Verify Setup

Test each endpoint:

```bash
# Test main app
curl -I https://mltrainer.windfuhr.net

# Test API
curl https://api.mltrainer.windfuhr.net/health

# Test monitor
curl https://monitor.mltrainer.windfuhr.net/status
```

## 🚨 Troubleshooting

### DNS Not Resolving
- Wait 24-48 hours for full propagation
- Check with: `nslookup mltrainer.windfuhr.net`
- Verify records in DNS panel

### SSL Certificate Errors
- Ensure domains are verified in Modal
- Check Modal dashboard for certificate status
- May take up to 10 minutes after DNS setup

### 404 Errors
- Verify Modal app is deployed
- Check endpoint labels match DNS names
- Ensure Modal workspace name is correct

## 📱 Mobile App Configuration

If you create a mobile app, use these URLs:

```javascript
const API_BASE = 'https://api.mltrainer.windfuhr.net';
const WEB_APP = 'https://mltrainer.windfuhr.net';
```

## 🎯 Final URLs

Once complete, your services will be available at:

- **Main App**: https://mltrainer.windfuhr.net
- **API**: https://api.mltrainer.windfuhr.net
- **Monitor**: https://monitor.mltrainer.windfuhr.net
- **Docs**: https://docs.mltrainer.windfuhr.net (optional)

## 📧 Email Configuration (Optional)

To send emails from mltrainer@windfuhr.net:

| Type | Name | Value | Priority |
|------|------|-------|----------|
| MX | mltrainer | `mail.windfuhr.net` | 10 |
| TXT | mltrainer | `v=spf1 include:modal.com ~all` | - |

---

**Need help?** Check Modal's custom domain docs or your DNS provider's guide.