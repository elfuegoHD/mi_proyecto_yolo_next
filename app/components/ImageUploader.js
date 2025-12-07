"use client";

import { useState } from "react";
import * as tf from "@tensorflow/tfjs";

export default function ImageUploader({ model }) {
  const [imageSrc, setImageSrc] = useState(null);
  const [predictions, setPredictions] = useState([]);

  const handleImageUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (ev) => setImageSrc(ev.target.result);
    reader.readAsDataURL(file);

    if (model) {
      const img = new Image();
      img.src = URL.createObjectURL(file);
      img.onload = async () => {
        const tensor = tf.browser.fromPixels(img).resizeNearestNeighbor([640, 640]).expandDims(0).toFloat();
        const output = await model.executeAsync(tensor);
        console.log("Predicciones:", output);
        setPredictions(output);
      };
    }
  };

  return (
    <div>
      <input type="file" accept="image/*" onChange={handleImageUpload} />
      {imageSrc && <img src={imageSrc} alt="Uploaded" style={{ maxWidth: 400 }} />}
      {predictions.length > 0 && <pre>{JSON.stringify(predictions, null, 2)}</pre>}
    </div>
  );
}
