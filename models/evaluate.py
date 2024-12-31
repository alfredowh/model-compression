import torch

def evaluate(model, device, test_loader):
    model.eval()

    losses = 0.0
    total_predictions = 0
    true_predictions_top1 = 0
    true_predictions_top5 = 0
    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            # Compute loss
            loss = criterion(outputs, targets) / inputs.size(0)
            losses += loss.item()

            # Top-1 predictions
            _, predicted_top1 = torch.max(outputs, 1)
            batch_true_predictions_top1 = (predicted_top1 == targets).sum().item()
            true_predictions_top1 += batch_true_predictions_top1

            # Top-5 predictions
            _, predicted_top5 = torch.topk(outputs, 5, dim=1)
            batch_true_predictions_top5 = sum(
                [targets[i].item() in predicted_top5[i].tolist() for i in range(targets.size(0))]
            )
            true_predictions_top5 += batch_true_predictions_top5

            # Update total predictions
            batch_total_predictions = outputs.size(0)
            total_predictions += batch_total_predictions

            # Print batch metrics
            # print(
            #     f'Batch {batch_idx}, Loss: {loss:.4f}, '
            #     f'Accuracy@1: {batch_true_predictions_top1 / batch_total_predictions * 100:.2f}%, '
            #     f'Accuracy@5: {batch_true_predictions_top5 / batch_total_predictions * 100:.2f}%'
            # )

    # Compute overall accuracies
    accuracy_top1 = true_predictions_top1 / total_predictions
    accuracy_top5 = true_predictions_top5 / total_predictions

    return accuracy_top1, accuracy_top5, losses