"use client";

import { useState, useEffect, useRef } from "react";
import * as tf from "@tensorflow/tfjs";

interface Prediction {
  x_min: number;
  y_min: number;
  x_max: number;
  y_max: number;
  confidence: number;
  class_id: number;
  label: string;
}

export default function Home() {
  const [model, setModel] = useState<tf.GraphModel | null>(null);
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [mensaje, setMensaje] = useState<string>("");
  const [mensajeColor, setMensajeColor] = useState<string>("black");
  const imageRef = useRef<HTMLImageElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Cargar modelo al iniciar
  useEffect(() => {
    const cargarModelo = async () => {
      const m = await tf.loadGraphModel("/best_web_model/model.json");
      setModel(m);
      console.log("Modelo cargado ✅");
    };
    cargarModelo();
  }, []);

  // Procesar predicciones de YOLO
  const procesarPredicciones = (output: tf.Tensor): Prediction[] => {
    const datos = (output as any).arraySync()[0] as number[][];

    const resultados: Prediction[] = datos
      .filter((box) => {
        const confianza = box[4];
        const ancho = box[2] - box[0];
        const alto = box[3] - box[1];
        const area = ancho * alto;

        // Ajuste: menor umbral de confianza y área mínima más pequeña
        return confianza > 0.05 && area > 100;
      })
      .map((box) => ({
        x_min: box[0],
        y_min: box[1],
        x_max: box[2],
        y_max: box[3],
        confidence: box[4],
        class_id: box[5],
        label: "Araña Roja",
      }));

    return resultados;
  };

  // Dibujar boxes en canvas adaptado al tamaño visible
  const dibujarCanvas = (preds: Prediction[]) => {
    if (!canvasRef.current || !imageRef.current) return;
    const ctx = canvasRef.current.getContext("2d");
    if (!ctx) return;

    const img = imageRef.current;
    const scaleX = img.width / img.naturalWidth;
    const scaleY = img.height / img.naturalHeight;

    canvasRef.current.width = img.width;
    canvasRef.current.height = img.height;
    ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);

    // dibujar imagen base
    ctx.drawImage(img, 0, 0, img.width, img.height);

    ctx.strokeStyle = "red";
    ctx.lineWidth = 3;
    ctx.font = `${16 * scaleX}px Arial`;
    ctx.fillStyle = "red";

    preds.forEach((p) => {
      const x = p.x_min * img.width;
      const y = p.y_min * img.height;
      const w = (p.x_max - p.x_min) * img.width;
      const h = (p.y_max - p.y_min) * img.height;

      ctx.strokeRect(x, y, w, h);
      ctx.fillText(
        `${p.label} ${(p.confidence * 100).toFixed(1)}%`,
        x,
        y > 20 ? y - 5 : y + 20
      );
    });
  };

  // Ejecutar predicción
  const handlePredict = async () => {
    if (!model || !imageRef.current) return;

    setMensaje("Procesando...");
    setMensajeColor("black");
    setPredictions([]);

    const img = tf.browser.fromPixels(imageRef.current).toFloat();
    const resized = tf.image.resizeBilinear(img, [640, 640]);
    const expanded = resized.expandDims(0);

    const output = model.execute(expanded) as tf.Tensor;
    const pred = procesarPredicciones(output);

    setPredictions(pred);
    dibujarCanvas(pred);

    if (pred.length > 0) {
      setMensaje(`⚠️ Se detectó Araña Roja (${pred.length} detección/es)`);
      setMensajeColor("red");
    } else {
      setMensaje("⚠️ Posible Araña Roja, revisar manualmente");
      setMensajeColor("orange");

      if (canvasRef.current && imageRef.current) {
        const ctx = canvasRef.current.getContext("2d");
        if (!ctx) return;
        ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
        ctx.drawImage(
          imageRef.current,
          0,
          0,
          imageRef.current.width,
          imageRef.current.height
        );
      }
    }

    tf.dispose([img, resized, expanded, output]);
  };

  return (
    <div
      style={{
        padding: "20px",
        fontFamily: "Arial, sans-serif",
        display: "flex",
        justifyContent: "center",
      }}
    >
      <div style={{ maxWidth: "750px", width: "100%" }}>
        <h1 style={{ textAlign: "center" }}>
          Detección de Araña Roja en Aguacate
        </h1>

        <input
          type="file"
          accept="image/*"
          onChange={(e) => {
            if (e.target.files) setImageFile(e.target.files[0]);
          }}
          style={{ marginTop: 20, display: "block", width: "100%" }}
        />

        {imageFile && (
          <div
            style={{
              marginTop: 20,
              border: "2px solid #ddd",
              borderRadius: 12,
              padding: 15,
              backgroundColor: "#f9f9f9",
              boxShadow: "0 4px 10px rgba(0,0,0,0.1)",
              position: "relative",
            }}
          >
            <div
              style={{
                textAlign: "center",
                fontWeight: "bold",
                marginBottom: 10,
              }}
            >
              Imagen a predecir
            </div>
            <div style={{ position: "relative" }}>
              <img
                ref={imageRef}
                src={URL.createObjectURL(imageFile)}
                alt="Imagen a predecir"
                style={{
                  width: "100%",
                  height: "auto",
                  display: "block",
                  borderRadius: 8,
                }}
                onLoad={() => {
                  if (canvasRef.current && imageRef.current) {
                    canvasRef.current.width = imageRef.current.width;
                    canvasRef.current.height = imageRef.current.height;
                    const ctx = canvasRef.current.getContext("2d");
                    if (!ctx) return;
                    ctx.clearRect(
                      0,
                      0,
                      canvasRef.current.width,
                      canvasRef.current.height
                    );
                    ctx.drawImage(
                      imageRef.current,
                      0,
                      0,
                      imageRef.current.width,
                      imageRef.current.height
                    );
                  }
                }}
              />
              <canvas
                ref={canvasRef}
                style={{
                  position: "absolute",
                  top: 0,
                  left: 0,
                  borderRadius: 8,
                  pointerEvents: "none",
                }}
              />
            </div>

            <button
              onClick={handlePredict}
              disabled={!model || !imageFile}
              style={{
                marginTop: 15,
                padding: "12px 25px",
                fontSize: 16,
                borderRadius: 6,
                backgroundColor: "#007bff",
                color: "white",
                border: "none",
                cursor: "pointer",
                fontWeight: "bold",
              }}
            >
              Detectar Araña Roja
            </button>

            {mensaje && (
              <div
                style={{
                  marginTop: 15,
                  padding: 12,
                  backgroundColor: "#f1f1f1",
                  borderRadius: 8,
                  fontWeight: "bold",
                  textAlign: "center",
                  color: mensajeColor,
                  fontSize: 16,
                }}
              >
                {mensaje}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
