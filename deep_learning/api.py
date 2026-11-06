
from io import BytesIO

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from PIL import Image, UnidentifiedImageError

from deep_learning.retinex_inference import enhance_image

app = FastAPI(
    title="Low-Light Image Enhancement API",
    version="1.0.0"
)


@app.get("/")
def home():
    return {"message": "Low-Light Enhancement API is running"}


@app.post("/enhance")
async def enhance(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert("RGB")

        enhanced = enhance_image(image)

        output = BytesIO()
        enhanced.save(output, format="PNG")
        output.seek(0)

        return StreamingResponse(
            output,
            media_type="image/png",
            headers={"Content-Disposition": "attachment; filename=enhanced.png"}
        )

    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Invalid image file")

    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))
