# Træningsloop:
# Idx, data = enumerate(DataLoader)
# Epochs er iterationer gennem hele dataet (det gør man n gange)
# Derefter itererer man gennem sin dataloader
# Pytorch laver en ny batch hver gang, man får nye billeder 
# DataSets har nogle funktioner, som DataLoader kalder (DataLoader laver ting som shuffling
# Optimizer zero-grad regner gradienterne og fjerner tidligere gradienter
# Man udregner loss og laver backpropagation (loss.backward() ) på den
# Tag et step: optimizer.step()

# Vores loop er lidt mere kompliceret fordi vi har 4 modeller
# Kig på de første convolutional GAN
# Kig evt. på Johans github (det er gammelt, men der er et loop, der virker)

