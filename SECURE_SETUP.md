# 🔒 Secure Environment Setup Guide

## Prerequisites
- New API keys from OpenAI and Pinecone (old ones revoked)
- Strong database password
- GitHub Personal Access Token for force push

## Steps

### 1. Update Environment Variables
```bash
# Copy the example file
cp .env.example .env

# Edit with your new credentials
nano .env
```

Update `.env` with:
```bash
# Database Configuration
POSTGRES_USER=user
POSTGRES_PASSWORD=YOUR_STRONG_NEW_PASSWORD_HERE
POSTGRES_DB=compliance_db

# Connection String (update with your new password)
DATABASE_URL=postgresql://user:YOUR_STRONG_NEW_PASSWORD_HERE@localhost:5432/compliance_db

# API Keys (use your NEW keys here)
OPENAI_API_KEY=your_new_openai_key_here
PINECONE_API_KEY=your_new_pinecone_key_here
```

### 2. Update Docker Environment
The `docker-compose.yml` now uses environment variables, so it will pick up your new credentials automatically.

### 3. Force Push Clean Repository
```bash
# Configure GitHub authentication (if not already done)
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"

# Use GitHub CLI or configure PAT for HTTPS
# Then force push the cleaned repository
git push --force origin main
```

### 4. Security Verification
```bash
# Run the security checklist
./security_checklist.sh
```

### 5. Test Your Setup
```bash
# Test database connection
docker-compose up db -d

# Test your application
# (run your normal startup commands here)
```

## 🛡️ Security Best Practices Going Forward

1. **Never commit `.env` files**
2. **Use different passwords for different environments**
3. **Rotate API keys regularly (quarterly)**
4. **Monitor API usage for anomalies**
5. **Use secrets management solutions for production**

## 🚨 If You See Suspicious Activity

1. **Immediately rotate all API keys again**
2. **Check billing/usage on all services**
3. **Review application logs for unauthorized access**
4. **Consider changing database passwords**
5. **Contact support for affected services**
