#!/bin/bash
# OpenCode LRS Serverless Deployment Script
# Phase 5: Ecosystem Expansion - Cloud Deployment

set -e

echo "🚀 OpenCode LRS Serverless Deployment"
echo "====================================="

# Check prerequisites
echo "📋 Checking prerequisites..."
command -v serverless >/dev/null 2>&1 || { echo "❌ Serverless Framework not found. Install with: npm install -g serverless"; exit 1; }
command -v aws >/dev/null 2>&1 || { echo "❌ AWS CLI not found. Install from: https://aws.amazon.com/cli/"; exit 1; }

# Set deployment variables
STAGE=${STAGE:-dev}
REGION=${REGION:-us-east-1}

echo "🔧 Deployment Configuration:"
echo "   Stage: $STAGE"
echo "   Region: $REGION"

# Create deployment package
echo "📦 Creating deployment package..."
mkdir -p dist

# Copy source files
cp -r src dist/
cp ../lightweight_lrs.py dist/
cp ../performance_optimization.py dist/
cp ../precision_calibration.py dist/
cp ../multi_agent_coordination.py dist/

# Copy requirements
cp requirements.txt dist/

echo "✅ Deployment package created"

# Deploy to AWS
echo "☁️  Deploying to AWS Lambda..."
serverless deploy --stage $STAGE --region $REGION

if [ $? -eq 0 ]; then
    echo "🎉 Deployment successful!"
    echo ""
    echo "🌐 Service Endpoints:"
    serverless info --stage $STAGE --region $REGION | grep -E "(GET|POST|ServiceEndpoint)" | sed 's/.*: //' | while read -r line; do
        if [[ $line == *"http"* ]]; then
            echo "   API Gateway: $line"
        fi
    done

    echo ""
    echo "📊 Lambda Functions:"
    aws lambda list-functions --region $REGION --query "Functions[?starts_with(FunctionName, 'opencode-lrs-serverless-$STAGE')].FunctionName" --output text | tr '\t' '\n' | while read -r func; do
        echo "   • $func"
    done

    echo ""
    echo "🗂️  S3 Buckets:"
    aws s3 ls | grep "opencode-lrs-cache-$STAGE" | awk '{print "   • "$3}'

    echo ""
    echo "🗃️  DynamoDB Tables:"
    aws dynamodb list-tables --region $REGION --query "TableNames[?starts_with(@, 'opencode-lrs-results-$STAGE')]" --output text | tr '\t' '\n' | while read -r table; do
        echo "   • $table"
    done
else
    echo "❌ Deployment failed!"
    exit 1
fi

echo ""
echo "🎯 Next Steps:"
echo "1. Update your VS Code extension settings:"
echo "   \"opencode.lrs.serverUrl\": \"<API_GATEWAY_URL>\""
echo ""
echo "2. Test the deployment:"
echo "   curl -X GET \"<API_GATEWAY_URL>/health\""
echo ""
echo "3. Monitor performance:"
echo "   serverless logs --function analyze --tail --stage $STAGE --region $REGION"