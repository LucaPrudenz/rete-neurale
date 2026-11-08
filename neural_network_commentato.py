"""
Rete neurale da zero: riconoscimento di cifre scritte a mano.
Solo NumPy: niente TensorFlow, niente PyTorch.

Architettura:  64 ingressi (pixel 8x8)  ->  32 neuroni nascosti (ReLU)  ->  10 uscite (softmax)

Differenze rispetto alla versione da 16 righe (neural_network.py):
  - tiene da parte 297 immagini mai viste per misurare l'accuratezza in modo onesto
  - stampa loss e accuratezza ogni 50 epoche
  - sottrae il massimo prima di np.exp (stabilità numerica, il risultato non cambia)
"""
import numpy as np
from sklearn.datasets import load_digits

# 1) DATI ---------------------------------------------------------------
# X: 1797 immagini, ognuna è una riga di 64 numeri (pixel 8x8 con valori 0..16)
# y: l'etichetta corretta (la cifra da 0 a 9)
X, y = load_digits(return_X_y=True)
X = X / 16                      # normalizziamo i pixel tra 0 e 1
Y = np.eye(10)[y]               # one-hot: 3 -> [0,0,0,1,0,0,0,0,0,0]

# Allenamento sulle prime 1500 immagini, test sulle altre 297 (mai viste)
X_tr, Y_tr, y_tr = X[:1500], Y[:1500], y[:1500]
X_te, y_te = X[1500:], y[1500:]

# 2) PARAMETRI ----------------------------------------------------------
np.random.seed(0)               # stessi numeri casuali a ogni esecuzione
lr = 0.5                        # learning rate: ampiezza di ogni correzione
W1 = np.random.randn(64, 32) * 0.1   # pesi ingresso -> nascosto (piccoli e casuali)
b1 = np.zeros(32)                    # bias del livello nascosto
W2 = np.random.randn(32, 10) * 0.1   # pesi nascosto -> uscita
b2 = np.zeros(10)                    # bias del livello d'uscita
# Totale: 64*32 + 32 + 32*10 + 10 = 2410 parametri da imparare


def forward(X):
    """Dalla matrice di immagini alle probabilità per ogni cifra."""
    h = np.maximum(0, X @ W1 + b1)                 # somma pesata + ReLU
    z = h @ W2 + b2                                # punteggi (logit)
    e = np.exp(z - z.max(1, keepdims=True))        # esponenziale (stabile)
    p = e / e.sum(1, keepdims=True)                # softmax: probabilità che sommano a 1
    return h, p


# 3) ALLENAMENTO --------------------------------------------------------
for epoch in range(300):
    h, p = forward(X_tr)

    # Errore in uscita: gradiente della cross-entropy rispetto ai logit
    d2 = (p - Y_tr) / len(X_tr)
    # Backpropagation: l'errore torna al livello nascosto attraverso W2;
    # (h > 0) è la derivata della ReLU: passa solo dai neuroni che erano accesi
    d1 = (d2 @ W2.T) * (h > 0)

    # Discesa del gradiente: ogni peso fa un piccolo passo "in discesa"
    W2 -= lr * h.T @ d2;    b2 -= lr * d2.sum(0)
    W1 -= lr * X_tr.T @ d1; b1 -= lr * d1.sum(0)

    if epoch % 50 == 0 or epoch == 299:
        loss = -np.mean(np.log(p[np.arange(len(y_tr)), y_tr] + 1e-12))
        acc = (p.argmax(1) == y_tr).mean()
        print(f"epoca {epoch:3d}   loss {loss:.3f}   accuratezza train {acc:.1%}")

# 4) VALUTAZIONE SU IMMAGINI MAI VISTE ----------------------------------
_, p_te = forward(X_te)
print(f"\nAccuratezza su 297 cifre mai viste: {(p_te.argmax(1) == y_te).mean():.1%}")
