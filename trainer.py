from tqdm import tqdm
EPOCHS = 10
def train_model(model, optimizer, loss_function, train_loader, device, epochs=10):
    model.train()
    for epoch in range(epochs):
        train_loader_with_progress = tqdm(
            train_loader,
            desc=f'Epoch {epoch+1}/{epochs}'
        )
        for inputs, labels in train_loader_with_progress:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loader_with_progress.set_postfix(
                loss=f'{loss.item():.4f}'
            )
    print("Finished Training")
    return model
