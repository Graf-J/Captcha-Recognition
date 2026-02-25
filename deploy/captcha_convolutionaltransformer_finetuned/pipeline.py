from transformers import Pipeline
import torch

class CaptchaPipeline(Pipeline):
    def _sanitize_parameters(self, **kwargs):
        return {}, {}, {}

    def preprocess(self, image):
        return self.processor(image)

    def _forward(self, model_inputs):
        with torch.no_grad():
            outputs = self.model(model_inputs["pixel_values"])
        return outputs

    def postprocess(self, model_outputs):
        logits = model_outputs.logits
        prediction = self.processor.batch_decode(logits)[0]
        return {"prediction": prediction}