import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import os

if not os.path.exists("gifs"):
    os.makedirs("gifs")


def merge_sort(arr):
    def merge_sort_recursive(arr, start_idx=0, transcript=None):
        if transcript is None:
            transcript = []

        if len(arr) > 1:
            mid = len(arr) // 2
            L = arr[:mid]
            R = arr[mid:]

            transcript.append(f"Split array into L={L} and R={R}")
            yield arr[
                :
            ], None, None, f"Dividing array: L={L}, R={R}", "split", transcript[:]

            yield from merge_sort_recursive(L, start_idx, transcript)
            yield from merge_sort_recursive(R, start_idx + mid, transcript)

            i = j = k = 0
            transcript.append("Begin merging sorted subarrays")

            while i < len(L) and j < len(R):
                transcript.append(f"Compare L[{i}]={L[i]} with R[{j}]={R[j]}")
                if L[i] <= R[j]:
                    arr[k] = L[i]
                    transcript.append(
                        f"Take {L[i]} from left array, place at position {k}"
                    )
                    yield arr[
                        :
                    ], start_idx + k, None, f"Merging: Place {L[i]} from left", "merge", transcript[
                        :
                    ]
                    i += 1
                else:
                    arr[k] = R[j]
                    transcript.append(
                        f"Take {R[j]} from right array, place at position {k}"
                    )
                    yield arr[
                        :
                    ], start_idx + k, None, f"Merging: Place {R[j]} from right", "merge", transcript[
                        :
                    ]
                    j += 1
                k += 1

            while i < len(L):
                arr[k] = L[i]
                transcript.append(f"Copy remaining left element {L[i]} to position {k}")
                yield arr[
                    :
                ], start_idx + k, None, f"Copy remaining {L[i]} from left", "place", transcript[
                    :
                ]
                i += 1
                k += 1

            while j < len(R):
                arr[k] = R[j]
                transcript.append(
                    f"Copy remaining right element {R[j]} to position {k}"
                )
                yield arr[
                    :
                ], start_idx + k, None, f"Copy remaining {R[j]} from right", "place", transcript[
                    :
                ]
                j += 1
                k += 1

            transcript.append(f"Merged result: {arr}")

    yield from merge_sort_recursive(arr.copy())


def bubble_sort(arr):
    n = len(arr)
    transcript = ["Starting Bubble Sort"]

    for i in range(n):
        transcript.append(f"Pass {i+1}: Largest {n-i} elements will bubble up")
        swapped = False

        for j in range(0, n - i - 1):
            transcript.append(
                f"Compare positions {j} and {j+1}: {arr[j]} vs {arr[j+1]}"
            )
            yield arr[
                :
            ], j, j + 1, f"Compare {arr[j]} and {arr[j+1]}", "compare", transcript[:]

            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True
                transcript.append(f"Swap! {arr[j]} and {arr[j+1]} exchanged positions")
                yield arr[
                    :
                ], j, j + 1, f"SWAP: {arr[j]} <-> {arr[j+1]}", "swap", transcript[:]
            else:
                transcript.append(f"No swap needed, {arr[j]} <= {arr[j+1]}")

        if not swapped:
            transcript.append("No swaps in this pass - array is sorted!")
            break
        else:
            transcript.append(
                f"Pass {i+1} complete, element {arr[n-i-1]} in final position"
            )

    transcript.append("Bubble Sort completed successfully")
    yield arr[:], None, None, "Bubble Sort Complete", "done", transcript[:]


def insertion_sort(arr):
    transcript = [
        "Starting Insertion Sort - build sorted portion one element at a time"
    ]

    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        transcript.append(f"Insert element at position {i}: key = {key}")
        transcript.append(f"Sorted portion: {arr[:i]}, Unsorted: {arr[i:]}")
        yield arr[
            :
        ], i, None, f"Key = {key}, finding insertion point", "key", transcript[:]

        transcript.append(
            f"Compare key {key} with sorted elements, moving right to left"
        )
        while j >= 0 and arr[j] > key:
            transcript.append(f"Shift {arr[j]} right (from pos {j} to {j+1})")
            arr[j + 1] = arr[j]
            yield arr[:], j, j + 1, f"Shift {arr[j+1]} right", "shift", transcript[:]
            j -= 1

        arr[j + 1] = key
        transcript.append(f"Insert key {key} at position {j+1}")
        transcript.append(f"Sorted portion now: {arr[:i+1]}")
        yield arr[:], j + 1, i, f"Insert {key} at position {j+1}", "insert", transcript[
            :
        ]

    transcript.append("Insertion Sort completed - entire array is sorted")
    yield arr[:], None, None, "Insertion Sort Complete", "done", transcript[:]


def selection_sort(arr):
    n = len(arr)
    transcript = [
        "Starting Selection Sort - find minimum and place in correct position"
    ]

    for i in range(n):
        min_idx = i
        transcript.append(f"Finding minimum element from position {i} to {n-1}")
        transcript.append(f"Sorted: {arr[:i]}, Unsorted: {arr[i:]}")
        yield arr[
            :
        ], i, min_idx, f"Search for minimum starting from position {i}", "search", transcript[
            :
        ]

        for j in range(i + 1, n):
            transcript.append(f"Check if {arr[j]} < current minimum {arr[min_idx]}")
            yield arr[
                :
            ], min_idx, j, f"Compare {arr[j]} with current min {arr[min_idx]}", "compare", transcript[
                :
            ]

            if arr[j] < arr[min_idx]:
                min_idx = j
                transcript.append(
                    f"New minimum found: {arr[min_idx]} at position {min_idx}"
                )
                yield arr[
                    :
                ], i, min_idx, f"New minimum: {arr[min_idx]}", "new_min", transcript[:]

        if min_idx != i:
            transcript.append(
                f"Swap minimum {arr[min_idx]} with element at position {i}"
            )
            arr[i], arr[min_idx] = arr[min_idx], arr[i]
            yield arr[
                :
            ], i, min_idx, f"Swap {arr[min_idx]} with {arr[i]}", "swap", transcript[:]
        else:
            transcript.append(f"Element {arr[i]} already in correct position")
            yield arr[
                :
            ], i, None, f"{arr[i]} already in correct position", "correct", transcript[
                :
            ]

        transcript.append(f"Position {i} now contains correct element: {arr[i]}")

    transcript.append("Selection Sort completed - all elements in correct positions")
    yield arr[:], None, None, "Selection Sort Complete", "done", transcript[:]


def quick_sort(arr, low=0, high=None, transcript=None):
    if transcript is None:
        transcript = ["Starting Quick Sort - divide and conquer approach"]
    if high is None:
        high = len(arr) - 1

    def partition(low, high):
        pivot = arr[high]
        i = low - 1
        transcript.append(f"Partitioning subarray from {low} to {high}")
        transcript.append(f"Choose pivot: {pivot} (rightmost element)")
        yield arr[:], high, None, f"Pivot = {pivot}", "pivot", transcript[:]

        transcript.append("Scan array, move elements <= pivot to left side")
        for j in range(low, high):
            transcript.append(f"Compare {arr[j]} with pivot {pivot}")
            yield arr[
                :
            ], j, high, f"Compare {arr[j]} with pivot {pivot}", "compare", transcript[:]

            if arr[j] <= pivot:
                i += 1
                if i != j:
                    transcript.append(
                        f"Move {arr[j]} to left partition (swap pos {i} and {j})"
                    )
                    arr[i], arr[j] = arr[j], arr[i]
                    yield arr[
                        :
                    ], i, j, f"Move {arr[j]} to left partition", "swap", transcript[:]
                else:
                    transcript.append(f"{arr[j]} already in correct partition")

        arr[i + 1], arr[high] = arr[high], arr[i + 1]
        transcript.append(f"Place pivot {pivot} at position {i+1} (final position)")
        transcript.append(
            f"Left partition: {arr[low:i+1]}, Pivot: {pivot}, Right: {arr[i+2:high+1]}"
        )
        yield arr[
            :
        ], i + 1, high, f"Pivot {pivot} in final position", "pivot_final", transcript[:]
        return i + 1

    if low < high:
        pi = yield from partition(low, high)
        transcript.append(f"Recursively sort left subarray: positions {low} to {pi-1}")
        yield from quick_sort(arr, low, pi - 1, transcript)
        transcript.append(
            f"Recursively sort right subarray: positions {pi+1} to {high}"
        )
        yield from quick_sort(arr, pi + 1, high, transcript)

    if low == 0 and high == len(arr) - 1:
        transcript.append("Quick Sort completed successfully")
        yield arr[:], None, None, "Quick Sort Complete", "done", transcript[:]


def heap_sort(arr):
    n = len(arr)
    transcript = ["Starting Heap Sort - first build max heap, then extract elements"]

    def heapify(n, i):
        largest = i
        l = 2 * i + 1  # left
        r = 2 * i + 2  # right

        transcript.append(f"Heapify node {i}: check children {l} and {r}")

        if l < n and arr[l] > arr[largest]:
            largest = l
            transcript.append(f"Left child {arr[l]} > parent {arr[i]}")
        if r < n and arr[r] > arr[largest]:
            largest = r
            transcript.append(f"Right child {arr[r]} is largest so far")

        if largest != i:
            transcript.append(f"Swap parent {arr[i]} with largest child {arr[largest]}")
            arr[i], arr[largest] = arr[largest], arr[i]
            yield arr[
                :
            ], i, largest, f"Heapify: swap {arr[largest]} with {arr[i]}", "heap", transcript[
                :
            ]
            yield from heapify(n, largest)

    # max heap
    transcript.append("Phase 1: Build max heap from unsorted array")
    for i in range(n // 2 - 1, -1, -1):
        transcript.append(f"Heapify subtree rooted at index {i}")
        yield from heapify(n, i)

    transcript.append("Max heap built successfully - largest element at root")

    # Extract
    transcript.append("Phase 2: Extract maximum elements one by one")
    for i in range(n - 1, 0, -1):
        transcript.append(f"Extract maximum {arr[0]}, place at position {i}")
        arr[0], arr[i] = arr[i], arr[0]
        yield arr[
            :
        ], 0, i, f"Extract max {arr[i]} to sorted position", "extract", transcript[:]

        transcript.append(f"Restore heap property for remaining {i} elements")
        yield from heapify(i, 0)

    transcript.append("Heap Sort completed - array fully sorted")
    yield arr[:], None, None, "Heap Sort Complete", "done", transcript[:]


def shell_sort(arr):
    n = len(arr)
    transcript = ["Starting Shell Sort - insertion sort with decreasing gaps"]

    gap = n // 2
    while gap > 0:
        transcript.append(f"Using gap size: {gap}")
        transcript.append(f"Perform gap-{gap} insertion sort on subarrays")
        yield arr[:], None, None, f"Gap size: {gap}", "gap", transcript[:]

        for i in range(gap, n):
            temp = arr[i]
            j = i
            transcript.append(f"Insert element {temp} at position {i} with gap {gap}")

            while j >= gap and arr[j - gap] > temp:
                transcript.append(
                    f"Shift {arr[j - gap]} from pos {j-gap} to {j} (gap {gap})"
                )
                arr[j] = arr[j - gap]
                yield arr[
                    :
                ], j, j - gap, f"Gap-{gap} shift: move {arr[j]}", "shift", transcript[:]
                j -= gap

            arr[j] = temp
            if j != i:
                transcript.append(f"Place {temp} at position {j}")
                yield arr[
                    :
                ], j, i, f"Place {temp} at gap position {j}", "place", transcript[:]

        transcript.append(f"Gap-{gap} sorting complete")
        gap //= 2

    transcript.append("Shell Sort completed - final gap-1 pass done")
    yield arr[:], None, None, "Shell Sort Complete", "done", transcript[:]


def counting_sort(arr):
    if not arr:
        return

    transcript = ["Starting Counting Sort - count occurrences, then place in order"]
    max_val = max(arr)
    min_val = min(arr)

    transcript.append(f"Range: {min_val} to {max_val}")
    transcript.append("Phase 1: Count frequency of each value")
    yield arr[
        :
    ], None, None, f"Count values from {min_val} to {max_val}", "count", transcript[:]

    range_of_elements = max_val - min_val + 1
    count = [0] * range_of_elements
    output = [0] * len(arr)

    for i in range(len(arr)):
        count[arr[i] - min_val] += 1
        transcript.append(f"Count occurrence of value {arr[i]}")
        yield arr[:], i, None, f"Counting value {arr[i]}", "counting", transcript[:]

    transcript.append("Phase 2: Calculate cumulative counts for positions")
    for i in range(1, len(count)):
        count[i] += count[i - 1]

    transcript.append("Phase 3: Place elements in sorted order")
    for i in range(len(arr) - 1, -1, -1):
        output[count[arr[i] - min_val] - 1] = arr[i]
        count[arr[i] - min_val] -= 1
        arr[:] = output[:]
        transcript.append(f"Place {arr[i]} in its final sorted position")
        yield arr[:], count[
            arr[i] - min_val
        ], None, f"Place {arr[i]} in sorted position", "place", transcript[:]

    transcript.append("Counting Sort completed - all elements counted and placed")
    yield arr[:], None, None, "Counting Sort Complete", "done", transcript[:]


sorting_algorithms = {
    "Bubble Sort": bubble_sort,
    "Insertion Sort": insertion_sort,
    "Selection Sort": selection_sort,
    "Quick Sort": quick_sort,
    "Heap Sort": heap_sort,
    "Shell Sort": shell_sort,
    "Counting Sort": counting_sort,
    "Merge Sort": merge_sort,
}


def get_color_scheme(action_type, is_highlight_1=False, is_highlight_2=False):
    """Get colors based on action type and highlighting"""
    color_schemes = {
        "compare": ("#FF6B6B", "#4ECDC4"),
        "swap": ("#FF8E53", "#FF6B9D"),
        "pivot": ("#9B59B6", "#3498DB"),
        "key": ("#E74C3C", "#F39C12"),
        "merge": ("#2ECC71", "#E67E22"),
        "shift": ("#F1C40F", "#8E44AD"),
        "place": ("#1ABC9C", "#34495E"),
        "extract": ("#E74C3C", "#95A5A6"),
        "heap": ("#9B59B6", "#F39C12"),
        "search": ("#3498DB", "#2ECC71"),
        "new_min": ("#E74C3C", "#F39C12"),
        "correct": ("#2ECC71", None),
        "gap": ("#8E44AD", "#F39C12"),
        "count": ("#3498DB", "#E74C3C"),
        "counting": ("#1ABC9C", "#E67E22"),
        "split": ("#9B59B6", "#F1C40F"),
        "done": ("#27AE60", None),
        "default": ("#74B9FF", "#FDCB6E"),
    }

    base_color = "#DDA0DD"
    scheme = color_schemes.get(action_type, color_schemes["default"])

    if is_highlight_1:
        return scheme[0]
    elif is_highlight_2:
        return scheme[1] if scheme[1] else scheme[0]
    else:
        return base_color


def visualize_sort(name, sort_func, arr):
    generator = sort_func(arr.copy())

    # Collect all frames
    frames = list(generator)
    if not frames:
        print(f"Warning: No frames generated for {name}")
        return

    # Set up figure
    plt.rcParams["font.family"] = "monospace"
    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor("#0D1117")

    gs = fig.add_gridspec(
        2, 2, height_ratios=[3, 1], width_ratios=[2, 1], hspace=0.3, wspace=0.2
    )

    ax_main = fig.add_subplot(gs[0, :])
    ax_main.set_facecolor("#161B22")
    ax_main.set_title(f"{name}", fontsize=18, fontweight="bold", color="white", pad=20)
    ax_main.set_ylim(0, max(arr) * 1.15)
    ax_main.set_xlim(-0.5, len(arr) - 0.5)
    ax_main.axis("off")
    ax_transcript = fig.add_subplot(gs[1, :])
    ax_transcript.set_facecolor("#0D1117")
    ax_transcript.axis("off")
    ax_transcript.set_xlim(0, 1)
    ax_transcript.set_ylim(0, 1)

    bars = ax_main.bar(
        range(len(arr)),
        arr,
        color="#DDA0DD",
        alpha=0.8,
        edgecolor="white",
        linewidth=1.5,
    )

    value_labels = []
    for i, bar in enumerate(bars):
        label = ax_main.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(arr) * 0.02,
            str(arr[i]),
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            color="white",
        )
        value_labels.append(label)

    # Status displays
    narration = ax_main.text(
        0.5,
        -0.08,
        "",
        ha="center",
        va="center",
        transform=ax_main.transAxes,
        fontsize=12,
        color="white",
        bbox=dict(
            facecolor="#21262D",
            edgecolor="#30363D",
            boxstyle="round,pad=0.6",
            alpha=0.9,
        ),
    )

    step_counter = ax_main.text(
        0.02,
        0.95,
        "",
        transform=ax_main.transAxes,
        fontsize=11,
        color="#7C3AED",
        fontweight="bold",
    )

    # Transcript text areas
    transcript_texts = []
    for i in range(6):  # Show last 6 steps
        text_obj = ax_transcript.text(
            0.02,
            0.85 - i * 0.14,
            "",
            transform=ax_transcript.transAxes,
            fontsize=9,
            color="#E5E7EB",
            fontfamily="monospace",
        )
        transcript_texts.append(text_obj)

    # Transcript title
    ax_transcript.text(
        0.02,
        0.95,
        "Algorithm Transcript:",
        transform=ax_transcript.transAxes,
        fontsize=11,
        color="#60A5FA",
        fontweight="bold",
    )

    def update(frame_idx):
        if frame_idx >= len(frames):
            return

        data = frames[frame_idx]

        if len(data) == 6:
            A, i, j, action, action_type, transcript = data
        else:
            A, i, j, action = data[:4]
            action_type = "default"
            transcript = []

        # Update bars and labels
        for k, (bar, label) in enumerate(zip(bars, value_labels)):
            bar.set_height(A[k])

            is_highlight_1 = i is not None and k == i
            is_highlight_2 = j is not None and k == j

            color = get_color_scheme(action_type, is_highlight_1, is_highlight_2)
            bar.set_color(color)

            if is_highlight_1 or is_highlight_2:
                bar.set_alpha(0.9)
                bar.set_linewidth(3)
                bar.set_edgecolor("white")
            else:
                bar.set_alpha(0.7)
                bar.set_linewidth(1.5)
                bar.set_edgecolor("#CCCCCC")

            label.set_position(
                (bar.get_x() + bar.get_width() / 2, bar.get_height() + max(arr) * 0.02)
            )
            label.set_text(str(A[k]))

        # Update displays
        narration.set_text(action)
        step_counter.set_text(f"Step {frame_idx + 1}/{len(frames)}")

        # Update transcript - show last 6 entries
        visible_transcript = transcript[-6:] if len(transcript) > 6 else transcript

        for i, text_obj in enumerate(transcript_texts):
            if i < len(visible_transcript):
                # Add step numbers to transcript entries
                step_num = len(transcript) - len(visible_transcript) + i + 1
                text_obj.set_text(f"{step_num:2d}. {visible_transcript[i]}")
                # Highlight current step
                if i == len(visible_transcript) - 1:
                    text_obj.set_color("#60A5FA")
                else:
                    text_obj.set_color("#E5E7EB")
            else:
                text_obj.set_text("")

    # Create animation
    ani = animation.FuncAnimation(
        fig, update, frames=len(frames), interval=1200, repeat=True, blit=False
    )

    # Save as GIF
    try:
        filename = os.path.join("gifs", f"{name.replace(' ', '_').lower()}.gif")
        ani.save(
            filename,
            writer="pillow",
            fps=0.8,
            dpi=100,
            savefig_kwargs={"facecolor": "#0D1117", "edgecolor": "none"},
        )
        plt.close(fig)
        print(f"Created professional visualization: {filename}")
    except Exception as e:
        print(f"Error creating {name}: {e}")
        plt.close(fig)


def main():
    arr = [8, 3, 7, 1, 9, 2, 6, 4, 5]
    print("Creating professional sorting algorithm visualizations...")
    print("=" * 60)

    for name, sort_func in sorting_algorithms.items():
        try:
            print(f"Processing {name}...")
            visualize_sort(name, sort_func, arr)
        except Exception as e:
            print(f"Error with {name}: {e}")

    print("=" * 60)
    print("Professional visualizations complete!")


if __name__ == "__main__":
    main()
