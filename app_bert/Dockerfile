FROM public.ecr.aws/lambda/python:3.12

COPY app_bert/requirements.txt ${LAMBDA_TASK_ROOT}
RUN pip install --no-cache-dir -r requirements.txt

# Only the small files: each Lambda downloads its ONNX model from S3 at start-up
# (MODEL_S3, see infra/api_bert.tf). Both come from s3://$DATA_BUCKET/v1/models/bert-token/
COPY app_bert/model/tokenizer.json app_bert/model/lemma_map.json /opt/model/
# bert_data.py imports common.py, so both go next to app.py
COPY common.py bert_data.py app_bert/app.py ${LAMBDA_TASK_ROOT}/

CMD ["app.handler"]
