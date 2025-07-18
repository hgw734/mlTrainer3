# Modal Deployment Guide for mlTrainer

## Prerequisites

1. **Modal Account**: Sign up at https://modal.com
2. **Modal CLI**: Install Modal
   ```bash
   pip install modal
   modal token new
   ```

## Step 1: Set Up Secrets in Modal

Go to https://modal.com/secrets and create a new secret called `mltrainer3-secrets` with:

```json
{
  "POLYGON_API_KEY": "your-polygon-api-key",
  "FRED_API_KEY": "your-fred-api-key", 
  "ANTHROPIC_API_KEY": "your-anthropic-api-key"
}
```

## Step 2: Deploy to Modal

```bash
# From your mlTrainer3 directory
modal deploy modal_mltrainer_complete.py
```

## Step 3: Access Your mlTrainer System

After deployment, you'll get URLs like:
- **Chat Interface**: `https://YOUR-WORKSPACE-mltrainer-chat.modal.run/`
- **Recommendations API**: `https://YOUR-WORKSPACE-mltrainer-api-recommendations.modal.run/`
- **Portfolio API**: `https://YOUR-WORKSPACE-mltrainer-api-portfolio.modal.run/`

## Features on Modal

### 1. **Always Available**
- Accessible from anywhere (iPhone, laptop, etc.)
- No need to be on same WiFi
- Automatic HTTPS/SSL

### 2. **Scheduled Jobs**
- **Every 15 minutes**: Scans for new trading recommendations
- **Every 5 minutes**: Updates virtual portfolio positions

### 3. **Persistent Storage**
- Recommendations saved to Modal volumes
- Portfolio history tracked
- Chat history preserved

### 4. **Auto-Scaling**
- Handles multiple users
- Scales up during heavy usage
- Scales down to save costs

## Accessing from iPhone

1. Save the chat URL to your iPhone home screen:
   - Open Safari
   - Go to your Modal URL
   - Tap Share button
   - Select "Add to Home Screen"

2. It will work like a web app with:
   - Full mlTrainer chat functionality
   - Real-time recommendations
   - Portfolio tracking
   - Mobile-optimized interface

## Cost Estimation

Modal pricing (as of 2024):
- **Compute**: ~$0.00001/second for this app
- **Storage**: ~$0.10/GB/month
- **Estimated monthly cost**: $5-20 depending on usage

## Monitoring

Check your app status:
```bash
modal app list
modal app logs mltrainer-complete
```

## Updating

To update after code changes:
```bash
git pull
modal deploy modal_mltrainer_complete.py --force
```

## Troubleshooting

1. **Streamlit not loading**: Check logs with `modal app logs`
2. **API keys not working**: Verify secrets in Modal dashboard
3. **Recommendations not updating**: Check scheduled function logs

## Custom Domain (Optional)

You can add a custom domain:
1. Go to Modal dashboard
2. Navigate to your app
3. Add custom domain
4. Point your domain's CNAME to Modal

Example: `mltrainer.yourdomain.com`