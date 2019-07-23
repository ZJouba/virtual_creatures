def open_file():
    from tkinter import filedialog
    import tkinter as tk
    import pickle

    root = tk.Tk()
    root.attributes("-topmost", True)
    root.withdraw()
    filepath = filedialog.askopenfilename()

    if filepath:
        print('\nFile selected')
        return pickle.load(open(filepath, 'rb'))
    else:
        exit()
