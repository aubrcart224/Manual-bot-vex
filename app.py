from flask import Flask, request, jsonify
from transformers import LlamaTokenizer, LlamaForCausalLM


app = Flask(__name__)

#load trained model and tokenizer

tokenizer = LlamaTokenizer.from_pretrained("facebook/llama-large")
model = LlamaForCausalLM.from_pretrained("facebook/llama-large")

@app.route('/generate', methods=['POST'])
def chat(): 
    user_input = request.json['message']
    inputs = tokenizer(user_input, return_tensors='pt')
    outputs = model.generate(inputs['input_ids'])
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return jsonify({'message': response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)