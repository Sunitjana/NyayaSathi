#!/bin/bash
echo "========================================"
echo "  Starting Legal AI Full Stack..."
echo "========================================"
echo ""

cd "$(dirname "$0")/legal-rag"
/opt/anaconda3/bin/python3 -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!

cd "$(dirname "$0")/frontend"
npx next dev &
FRONTEND_PID=$!

sleep 4
echo ""
echo "========================================"
echo "  Backend:  http://localhost:8000"
echo "  Frontend: http://localhost:3000"
echo "========================================"
echo ""
echo "Press Ctrl+C to stop both servers"

trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit" INT TERM
wait
