# Rete neurale da zero

Materiale per il corso: una rete neurale che impara a riconoscere cifre scritte a mano, scritta solo con NumPy, più un'animazione da proiettare in aula in cui la stessa rete si allena davvero, dentro il browser.

| File | A cosa serve |
|---|---|
| `index.html` | L'animazione "a capitoli" da proiettare in aula. Non richiede installazioni. |
| `neural_network.py` | Le 16 righe originali, trascritte fedelmente. Accuratezza circa 98% sulle immagini usate per l'allenamento. |
| `neural_network_commentato.py` | La stessa rete commentata riga per riga in italiano, che tiene da parte 297 immagini mai viste per misurare l'accuratezza in modo onesto (circa 91%). |

---

## 1. Aprire l'animazione

**Dal computer (senza internet va bene lo stesso).**

1. Scarica il repository (su GitHub: pulsante verde **Code** → **Download ZIP**, poi estrai lo ZIP).
2. Fai doppio clic su `index.html`. Si apre nel browser predefinito.
3. Consigliati Chrome, Edge, Firefox o Safari aggiornati.

Senza connessione l'animazione funziona comunque: cambiano solo i caratteri tipografici, che vengono sostituiti da quelli di sistema.

**Da un link online.** Apri **https://lucaprudenz.github.io/rete-neurale/**: funziona anche da tablet e smartphone, e i corsisti possono riaprirla da casa.

## 2. Comandi

| Comando | Cosa fa |
|---|---|
| **Play / Pausa** oppure **barra spaziatrice** | Avvia o ferma l'animazione |
| **Frecce ← →** | Capitolo precedente / successivo |
| Clic su un **capitolo** nella barra | Salta direttamente a quel capitolo |
| Menu **velocità** (0,5× · 1× · 2×) | Rallenta o accelera, utile per le domande in aula |
| **Schermo intero** | Ingrandisce la scena per il proiettore (Esc per uscire) |
| **Cancella** / **Cifra di test** | Nel capitolo "Prova tu": pulisce il riquadro o carica una cifra vera dal test |

Suggerimento: se la barra spaziatrice sembra non rispondere, fai prima un clic su un punto vuoto della pagina (se l'ultimo clic è stato su un pulsante, la barra spaziatrice "preme" quel pulsante).

## 3. Come usarla in aula (traccia di circa 20 minuti)

| Capitolo | Durata | Cosa mostra | Spunto per i corsisti |
|---|---|---|---|
| **Intro** | 1 min | Il problema: leggere cifre scritte a mano | "Come spieghereste a un computer cos'è un 7?" |
| **1 · Pixel** | 2 min | Un'immagine 8×8 diventa 64 numeri tra 0 e 1 | Il computer non "vede", legge numeri |
| **2 · Pesi** | 2 min | Ogni linea è un peso, all'inizio casuale | 2.410 numeri da imparare in totale |
| **3 · Forward** | 3 min | Il segnale (ambra) va avanti: somma pesata, ReLU, softmax | Collegare alla riga `h = np.maximum(0, X @ W1 + b1)` |
| **4 · Backprop** | 3 min | L'errore (ciano) torna indietro e corregge i pesi | Cos'è un gradiente, detto semplice: "da che parte scendere" |
| **5 · Allenamento** | 3 min | 300 epoche dal vivo: la loss scende, l'accuratezza sale | Fermare a metà con Pausa e chiedere previsioni |
| **6 · Risultato** | 2 min | Circa 91% su cifre mai viste | Perché si misura su dati mai visti |
| **Prova tu** | libero | Disegna una cifra col mouse o col dito e la rete la legge | Far disegnare ai corsisti cifre "difficili" (un 4 aperto, un 7 col trattino) |

Sotto la scena, a sinistra, c'è il codice Python con la riga attiva evidenziata a ogni capitolo; a destra il grafico della loss. Così si può passare dall'animazione al codice senza cambiare finestra.

Ogni apertura della pagina riallena la rete da capo con pesi casuali: le percentuali possono variare di poco da una volta all'altra, ed è normale.

## 4. Eseguire gli script Python

Serve Python 3 con due librerie:

```bash
pip install numpy scikit-learn
python neural_network.py              # stampa l'accuratezza finale (circa 0.98)
python neural_network_commentato.py   # stampa loss e accuratezza ogni 50 epoche, poi il test (circa 91%)
```

In alternativa, gli script funzionano anche copiandoli in una cella di [Google Colab](https://colab.research.google.com), senza installare nulla.

## 5. Pubblicazione

La pagina è pubblicata gratuitamente con GitHub Pages dal branch `main`, cartella principale. Ogni modifica a `index.html` caricata su `main` va online da sola in un paio di minuti.
