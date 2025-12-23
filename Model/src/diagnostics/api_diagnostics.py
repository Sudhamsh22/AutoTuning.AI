from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any, Dict, Optional

from src.diagnostics.diagnoser import diagnose

app = FastAPI(title="Vehicle Diagnostics API", version="1.0")


class DiagnoseReq(BaseModel):
    query: Optional[str] = None
    message: Optional[str] = None
    vehicle_type: Optional[str] = "car"
    brand: Optional[str] = None
    model: Optional[str] = None
    year: Optional[int] = None
    mileage: Optional[int] = None
    priority: Optional[str] = None
    topk: int = 3


@app.post("/critical-diagnosis")
def critical_diagnosis(req: DiagnoseReq):
    try:
        query = req.query or req.message or req.problemDescription
        if not query:
            raise HTTPException(status_code=400, detail="Missing problem description")

        vt = (req.vehicle_type or "car").lower()
        if vt not in ("car", "bike"):
            vt = "car"

        # temporary: bike routed to car diagnoser
        if vt == "bike":
            vt = "car"

        results = diagnose(vt, query, topk=req.topk)

        if not results:
            reply = "I couldn’t find a clear diagnosis based on the description. Please add more details."
        else:
            top = results[0]
            causes = ", ".join(top["causes"][:3])
            reply = (
                f"Based on your description, a possible issue is **{top['failure']}**.\n\n"
                f"Common causes include: {causes}."
            )

        return {
            "reply": reply,
            "results": results,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
