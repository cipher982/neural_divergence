import torch
import torch.nn as nn
import torch.optim as optim

# Define a simple neural network
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Function to calculate cosine similarity
def cosine_similarity(v1, v2):
    v1_flat = v1.view(-1)
    v2_flat = v2.view(-1)
    return torch.dot(v1_flat, v2_flat) / (torch.norm(v1_flat) * torch.norm(v2_flat))

# Create and train the model
model = SimpleNet()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Dummy training data
X = torch.randn(100, 10)
y = torch.randn(100, 1)

# Train the model (just a few epochs for demonstration)
for epoch in range(10):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()

# Function to measure divergence
def measure_divergence(model, input_data, perturbation_scale=0.01):
    model.eval()
    with torch.no_grad():
        # Original output
        original_output = model(input_data)
        
        # Perturbed input
        perturbed_input = input_data + torch.randn_like(input_data) * perturbation_scale
        perturbed_output = model(perturbed_input)
        
        # Calculate divergence (using cosine similarity)
        divergence = 1 - cosine_similarity(original_output, perturbed_output)
        
        return divergence.item()

# Test the divergence measurement
test_input = torch.randn(1, 10)
divergence = measure_divergence(model, test_input)

print(f"Divergence: {divergence}")

# Measure divergence for different perturbation scales
perturbation_scales = [0.001, 0.01, 0.1, 1.0]
for scale in perturbation_scales:
    div = measure_divergence(model, test_input, perturbation_scale=scale)
    print(f"Divergence at scale {scale}: {div}")