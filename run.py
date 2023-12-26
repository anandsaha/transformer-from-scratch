import transformer

print("Testing Transformer", "=" * 20)
transformer = transformer.build_transformer(30000, 30000, 1000, 1000)

counter = 0
for p in transformer.parameters():
    counter += 1
print("Number of parameters: ", counter)