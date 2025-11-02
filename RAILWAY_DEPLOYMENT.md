# 🚂 Railway Deployment Guide

## Why Railway is Better for This Project

| Feature | Railway | Render (Free) |
|---------|---------|---------------|
| **RAM** | 512MB (upgradable to 8GB) | 512MB |
| **Build Time** | Faster | Slower |
| **Spin-down** | No automatic spin-down | Spins down after 15min |
| **Startup** | Faster cold starts | Slow cold starts |
| **Free Credits** | $5/month trial | Limited hours/month |
| **Better for ML** | ✅ Yes | ❌ Struggles |

**Recommendation: Use Railway!** 🚂

---

## 🚀 Deploy to Railway (3 Steps)

### **Step 1: Push to GitHub**

```bash
git add .
git commit -m "Add Railway configuration"
git push
```

### **Step 2: Deploy on Railway**

1. Go to [railway.app](https://railway.app)
2. Sign in with GitHub
3. Click **"New Project"**
4. Choose **"Deploy from GitHub repo"**
5. Select your `face-detection` repository
6. Railway will automatically detect Python and deploy!

### **Step 3: Configure (Optional)**

Railway auto-detects most settings, but you can customize:

**In Railway Dashboard → Settings:**
- **Start Command**: Already configured in `railway.json`
- **Custom Domain**: Add your own domain (optional)
- **Environment Variables**: Auto-configured in `nixpacks.toml`

---

## ⚙️ What's Already Configured

### **Files for Railway:**
✅ `railway.json` - Railway-specific configuration
✅ `nixpacks.toml` - Build and environment settings
✅ `requirements.txt` - Python dependencies
✅ `app.py` - Optimized for low memory usage

### **Automatic Configuration:**
✅ Python 3.11
✅ Gunicorn with 1 worker (memory-optimized)
✅ 300-second timeout (for ML model loading)
✅ TensorFlow optimizations
✅ Auto-restart on failure

---

## 🎯 Railway vs Render Settings

### **Render Command (what you had):**
```bash
gunicorn -w 2 -b 0.0.0.0:$PORT app:app --timeout 120 --log-level info
```
❌ **Problem**: 2 workers × heavy TensorFlow = OUT OF MEMORY

### **Railway Command (optimized):**
```bash
gunicorn -w 1 -b 0.0.0.0:$PORT app:app --timeout 300 --worker-class sync --max-requests 100 --log-level info --preload
```
✅ **Better**:
- 1 worker = less memory
- 300s timeout = enough time to load model
- `--preload` = load model once, not per worker
- `--max-requests 100` = restart worker to free memory

---

## 🔧 Memory Optimizations Applied

### **1. Lazy Loading**
```python
# Model only loads when first request comes in
detector = None

def get_detector():
    global detector
    if detector is None:
        detector = EmotionDetector(model_path='face_model.h5')
    return detector
```

### **2. Single Worker**
- Uses only 1 Gunicorn worker instead of 2+
- Reduces memory footprint by 50%

### **3. Worker Recycling**
- `--max-requests 100` = worker restarts after 100 requests
- Prevents memory leaks

### **4. Preloading**
- `--preload` = load app before forking workers
- Model loads once, shared across requests

---

## 📊 Expected Performance

### **Free Tier (512MB RAM):**
- ✅ Can handle the model and app
- ✅ 1-2 concurrent users
- ✅ Response time: 2-5 seconds
- ⚠️ May still timeout on first load

### **If You Upgrade ($5-10/month):**
- ✅ 2GB RAM or more
- ✅ 5-10 concurrent users
- ✅ Faster response times
- ✅ No timeouts

---

## 🐛 Troubleshooting

### **Problem: Still Getting Timeouts**

**Solution 1: Add health check delay**
In Railway Dashboard → Settings → Health Check:
- Set **Start Probe Delay** to `120 seconds`

**Solution 2: Reduce model size**
```python
# In emotion_model.py, use a smaller model
# Or use model quantization
```

**Solution 3: Upgrade to Hobby Plan** ($5/month)
- 2GB RAM
- Much more reliable

### **Problem: Out of Memory**

**Check logs in Railway dashboard:**
```bash
# If you see "killed" or "signal 9"
# This means out of memory
```

**Solutions:**
1. Upgrade to Hobby plan (2GB RAM)
2. Use a smaller TensorFlow model
3. Optimize the model with quantization

### **Problem: Webcam Not Working**

**Railway automatically provides HTTPS** ✅
- Webcam requires HTTPS
- Railway gives you `https://your-app.up.railway.app`
- Should work automatically!

**If still not working:**
- Check browser camera permissions
- Try different browser (Chrome works best)
- Check Railway logs for errors

---

## 💰 Pricing

### **Trial Plan (Your Current):**
- $5 free credits
- Usage-based billing after credits
- ~500 hours of app runtime with trial credits
- Perfect for testing!

### **Hobby Plan ($5/month):**
- $5 credit included
- Pay only for usage above $5
- Better for production
- More reliable

### **Pro Plan ($20/month):**
- $20 credit included
- Priority support
- Better performance

**Recommendation**: Start with trial, upgrade if needed.

---

## 🔄 Updating Your Deployment

### **After Making Changes:**

```bash
git add .
git commit -m "Update app"
git push
```

Railway **automatically redeploys** when you push to GitHub! 🎉

---

## 📝 Environment Variables (Optional)

If you need to add custom settings:

**In Railway Dashboard → Variables:**
- Click **"New Variable"**
- Add custom environment variables

**Already configured** (in nixpacks.toml):
- `PYTHON_VERSION=3.11`
- `TF_CPP_MIN_LOG_LEVEL=2`
- `TF_ENABLE_ONEDNN_OPTS=0`

---

## 🌐 Your App URLs

After deployment, you'll get:
- **Railway URL**: `https://your-app-name.up.railway.app`
- **Main Page**: `https://your-app-name.up.railway.app/`
- **Webcam**: `https://your-app-name.up.railway.app/webcam`

Railway provides **automatic HTTPS** ✅

---

## ✅ Deployment Checklist

- [ ] Code pushed to GitHub
- [ ] `railway.json` added
- [ ] `nixpacks.toml` added
- [ ] `requirements.txt` has all dependencies
- [ ] `face_model.h5` in repository
- [ ] Railway account created
- [ ] Project deployed from GitHub
- [ ] App accessible via Railway URL
- [ ] Image upload tested
- [ ] Webcam tested (requires HTTPS)

---

## 🆚 Railway vs Render vs Heroku

| Feature | Railway | Render | Heroku |
|---------|---------|--------|--------|
| **Ease of Use** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Free Tier** | $5 credit | Limited hours | No free tier |
| **Auto-deploy** | ✅ Yes | ✅ Yes | ✅ Yes |
| **HTTPS** | ✅ Auto | ✅ Auto | ✅ Auto |
| **ML/AI Friendly** | ✅ Yes | ⚠️ Limited | ⚠️ Limited |
| **Cold Starts** | Fast | Slow | Medium |
| **Best For** | ML apps | Static sites | General apps |

**Winner for your project**: **Railway** 🏆

---

## 🎉 Summary

### **What to Do:**

1. **Push files to GitHub**:
   ```bash
   git add .
   git commit -m "Add Railway config"
   git push
   ```

2. **Deploy on Railway**:
   - Go to railway.app
   - New Project → Deploy from GitHub
   - Select your repo
   - Done!

3. **Test your app**:
   - Visit your Railway URL
   - Test image upload
   - Test webcam feature

### **Key Advantages:**
✅ No spin-down (unlike Render free tier)
✅ Faster deployments
✅ Better for ML/TensorFlow apps
✅ Automatic HTTPS
✅ Easy GitHub integration
✅ $5 free credits to start

---

**Your emotion detection app will work much better on Railway! 🚂🎉**
