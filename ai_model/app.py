from flask import Flask, request, jsonify
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from transformers import AutoTokenizer, AutoModel

class GCN(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, 16)
        self.conv2 = GCNConv(16, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

class DimensionAdapter(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DimensionAdapter, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)
    

protbert_tokenizer = AutoTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
protbert_model = AutoModel.from_pretrained("Rostlab/prot_bert")

# Load your GCN model
gcn_model = GCN(num_features=1280, num_classes=3)
gcn_model.load_state_dict(torch.load('model_state_dict.pth', map_location=torch.device('cpu')))
gcn_model.eval()

# Initialize the dimension adapter 
adapter = DimensionAdapter(1024, 1280)

def prepare_data(sequence):
    # Tokenize seq
    encoded_sequence = protbert_tokenizer(sequence, return_tensors="pt")
    # print(encoded_sequence)
    with torch.no_grad():
        outputs = protbert_model(**encoded_sequence)
        sequence_embedding = outputs.last_hidden_state[:, 0, :]
        # print("seq_emb", sequence_embedding)
        # increase diemsnions
        sequence_embedding_expanded = adapter(sequence_embedding)
        print("[seq-emb-exp]", sequence_embedding_expanded)
    # Dummy edge_index for a single node
    edge_index = torch.tensor([[0], [0]], dtype=torch.long)
    # data of type x: [1,1280], edge_index[2,1]
    return Data(x=sequence_embedding_expanded, edge_index=edge_index)
app = Flask(__name__)
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json['sequence']
        tensor_data = prepare_data(data)
        print("Tensor data", tensor_data)
        with torch.no_grad():
            out = gcn_model(tensor_data)
            print('----------------Out--------------',out)

            probabilities = torch.softmax(out, dim=1).tolist()
            # print("Print probability",probabilities)
            response = {'biological_process': probabilities[0][0], 
                        'cellular_component': probabilities[0][1],
                        'molecular_function': probabilities[0][2], 
                        'status': 'success'}

        return jsonify(response)
    except Exception as e:
        print(e)

if __name__ == '__main__':
    app.run(debug=True)
