#!/bin/bash

# Security Incident Response Checklist
# Execute these commands in order after completing manual API key rotation

echo "🔐 Compliance Chatbot - Security Incident Response"
echo "=================================================="

# Step 1: Verify clean repository
echo "Step 1: Verifying repository is clean..."
git log --all --full-history -- .env
if [ $? -eq 0 ]; then
    echo "❌ .env still found in history - git filter-repo may have failed"
else
    echo "✅ .env successfully removed from git history"
fi

# Step 2: Check for any remaining sensitive data
echo -e "\nStep 2: Scanning for any remaining sensitive data..."
grep -r "sk-proj-\|sk-[a-zA-Z0-9]\{48,\}\|pcsk_" --exclude-dir=.git --exclude="*.md" . || echo "✅ No API keys found in code"

# Step 3: Verify .gitignore is protecting .env
echo -e "\nStep 3: Checking .gitignore protection..."
if grep -q "^\.env$" .gitignore; then
    echo "✅ .gitignore properly excludes .env files"
else
    echo "❌ .env not properly excluded in .gitignore"
fi

# Step 4: Check environment variable usage
echo -e "\nStep 4: Verifying environment variable usage..."
grep -r "os\.getenv\|os\.environ" backend/ || echo "⚠️  Consider using environment variables"

echo -e "\n🚨 MANUAL ACTIONS REQUIRED:"
echo "1. Go to https://platform.openai.com/api-keys"
echo "2. Revoke key: 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"
echo "3. Generate new OpenAI API key"
echo "4. Go to https://app.pinecone.io/organizations/-/projects/-/keys"
echo "5. Revoke key: pcsk_zZeJW_Gkf6dFhH9NvRQJBVoMGExu44uejUzrNaMkWWvjnNSLsYPHPF4VFBRcMPExhFXfr"
echo "6. Generate new Pinecone API key"
echo "7. Update your .env file with new keys"
echo "8. Configure GitHub authentication (PAT) and run: git push --force origin main"

echo -e "\n📊 MONITORING COMMANDS:"
echo "# Check OpenAI usage:"
echo "curl -H 'Authorization: Bearer NEW_API_KEY' https://api.openai.com/v1/usage"
echo "# Monitor your applications logs for any errors"

echo -e "\n✅ Repository cleanup completed!"
echo "❌ API key rotation still required!"
