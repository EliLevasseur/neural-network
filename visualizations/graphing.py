import sys
from queue import Empty, Queue
from threading import Thread

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

plt.style.use("dark_background")

points = Queue()
input_finished = False


def read_training_output():
    global input_finished

    for line in sys.stdin:
        try:
            epoch_text, loss_text = line.strip().split(",")
            points.put((int(epoch_text), float(loss_text)))
        except ValueError:
            print(f"Skipping invalid training output: {line.rstrip()}", file=sys.stderr)

    input_finished = True


epochs = []
losses = []

fig, ax = plt.subplots()
loss_line, = ax.plot([], [], color="tab:blue", linewidth=2)

ax.set_xlabel("Epoch")
ax.set_ylabel("Binary cross-entropy loss")
ax.set_title("Training Loss vs. Epoch")
ax.grid(alpha=0.01)


def update(_frame):
    try:
        epoch, loss = points.get_nowait()
        epochs.append(epoch)
        losses.append(loss)
    except Empty:
        pass

    if epochs:
        loss_line.set_data(epochs, losses)
        ax.relim()
        ax.autoscale_view()

    if input_finished and points.empty():
        animation.event_source.stop()

    return (loss_line,)


reader = Thread(target=read_training_output, daemon=True)
reader.start()

animation = FuncAnimation(
    fig,
    update,
    interval=5,
    cache_frame_data=False,
)

plt.tight_layout()
plt.show()
