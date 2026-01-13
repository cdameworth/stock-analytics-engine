#!/bin/bash
# ============================================================
# Railway Setup Script for Stock Analytics Engine
# ============================================================
#
# This script sets up the Railway project with PostgreSQL
# and configures all services with the correct environment variables.
#
# Prerequisites:
#   - Railway CLI installed: npm install -g @railway/cli
#   - Railway account and logged in: railway login
#
# Usage:
#   ./setup_railway.sh
#
# ============================================================

set -e

echo "============================================================"
echo "Stock Analytics Engine - Railway Setup"
echo "============================================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if Railway CLI is installed
if ! command -v railway &> /dev/null; then
    echo -e "${RED}Error: Railway CLI not installed${NC}"
    echo "Install it with: npm install -g @railway/cli"
    exit 1
fi

# Check if logged in
if ! railway whoami &> /dev/null; then
    echo -e "${YELLOW}Not logged in to Railway. Running 'railway login'...${NC}"
    railway login
fi

echo -e "${GREEN}Railway CLI is ready${NC}"
echo ""

# Get project status
echo "Checking current project status..."
if railway status --json &> /dev/null; then
    echo -e "${GREEN}Already linked to a Railway project${NC}"
    railway status
else
    echo -e "${YELLOW}No project linked. Creating new project...${NC}"
    railway init
fi

echo ""
echo "============================================================"
echo "Step 1: Add PostgreSQL Database"
echo "============================================================"
echo ""

# Check if PostgreSQL exists
echo "Checking for existing PostgreSQL service..."
if railway status --json 2>/dev/null | grep -q "Postgres"; then
    echo -e "${GREEN}PostgreSQL already exists in this project${NC}"
else
    echo "Adding PostgreSQL database..."
    railway add postgres
    echo -e "${GREEN}PostgreSQL added successfully${NC}"
fi

echo ""
echo "============================================================"
echo "Step 2: Configure Environment Variables"
echo "============================================================"
echo ""

# Get services (this will show what services exist)
echo "Current services in project:"
railway status

echo ""
echo "Setting up DATABASE_URL references for each service..."
echo ""

# Configure api-service
echo "Configuring api-service..."
railway variables set -s api-service \
    DATABASE_URL='${{Postgres.DATABASE_URL}}' \
    2>/dev/null || echo "  Note: api-service may not exist yet"

# Configure model-tuning
echo "Configuring model-tuning..."
railway variables set -s model-tuning \
    DATABASE_URL='${{Postgres.DATABASE_URL}}' \
    ACCURACY_THRESHOLD="0.65" \
    MARKET_OUTPERFORMANCE_THRESHOLD="0.03" \
    SHARPE_RATIO_MIN="1.0" \
    MAX_DRAWDOWN_LIMIT="0.15" \
    CALIBRATION_ERROR_THRESHOLD="0.10" \
    RETRAINING_ACCURACY_THRESHOLD="0.50" \
    ENABLE_CONTINUOUS_MONITORING="true" \
    2>/dev/null || echo "  Note: model-tuning may not exist yet"

# Configure data-ingestion
echo "Configuring data-ingestion..."
railway variables set -s data-ingestion \
    DATABASE_URL='${{Postgres.DATABASE_URL}}' \
    2>/dev/null || echo "  Note: data-ingestion may not exist yet"

echo ""
echo -e "${GREEN}Environment variables configured${NC}"

echo ""
echo "============================================================"
echo "Step 3: Run Database Migration"
echo "============================================================"
echo ""

echo "To run the database migration, you need the DATABASE_URL."
echo ""
echo "Option 1: Run migration via Railway shell"
echo "  railway run python railway/scripts/migrate_postgres.py"
echo ""
echo "Option 2: Get DATABASE_URL and run locally"
echo "  railway variables -s Postgres"
echo "  export DATABASE_URL='<copy the URL>'"
echo "  python railway/scripts/migrate_postgres.py"
echo ""

echo "============================================================"
echo "Setup Complete!"
echo "============================================================"
echo ""
echo "Next steps:"
echo "  1. Run the database migration (see Step 3 above)"
echo "  2. Deploy services: railway up"
echo "  3. Check logs: railway logs -s api-service"
echo ""
echo "For more info, see: docs/ACCURACY_TRACKING_IMPROVEMENTS.md"
echo ""
