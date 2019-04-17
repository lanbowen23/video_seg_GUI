import os, glob
from sys import path
from tkinter.ttk import Label
from PIL import Image, ImageTk
import matplotlib

path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
matplotlib.use("TkAgg")


class AnimatedGIF(Label, object):
    def __init__(self, master, path_to_gif):
        self._master = master
        self._loc = 0

        im = Image.open(path_to_gif)
        self._frames = []
        i = 0
        try:
            while True:
                temp = im.copy()
                self._frames.append(ImageTk.PhotoImage(temp.convert('RGBA')))

                i += 1
                im.seek(i)
        except EOFError:
            pass

        self._len = len(self._frames)

        try:
            self._delay = im.info['duration']
        except:
            self._delay = 100

        self._callback_id = None

        super(AnimatedGIF, self).__init__(master, image=self._frames[0])

    def _run(self):
        self._loc += 1
        if self._loc == self._len:
            self._loc = 0

        self.configure(image=self._frames[self._loc])
        self._callback_id = self._master.after(self._delay, self._run)

    def pack(self, *args, **kwargs):
        self._run()
        super(AnimatedGIF, self).pack(*args, **kwargs)

    def grid(self, *args, **kwargs):
        self._run()
        super(AnimatedGIF, self).grid(*args, **kwargs)

    def place(self, *args, **kwargs):
        self._run()
        super(AnimatedGIF, self).place(*args, **kwargs)


if __name__ == "__main__":
    from tkinter import Tk, Button, Label
    from tkinter import filedialog
    from tkinter import messagebox
    import os

    root = Tk()

    root.title('Demo')
    root.geometry('1050x700')

    data_path = ""


    def getDataset():
        directory = filedialog.askdirectory()
        print(directory)
        root.update()
        split_path = directory.split('/')

        if 'youtube' in split_path[-1].lower():
            global data_path
            data_path = split_path[-1]
        else:
            messagebox.showerror("Error", "This directory is not test data")


    whole_model_path = os.path.join('Models', 'osmn.ckpt-300000')


    def predict():
        print("remove all previous file first")
        os.chdir("./")
        for file in glob.glob("*.gif"):
            os.remove(file)

        if data_path != "" and whole_model_path != "":
            import predict
            predict.predict(data_path, whole_model_path)
            predict.merge_masks(data_path, whole_model_path)
            predict.get_results(data_path)


    def show():
        save_dir = os.getcwd()
        os.chdir("./")

        count = 0
        for _ in glob.glob("*.gif"):
            gif1 = AnimatedGIF(root, save_dir + "/demo" + str(count) + ".gif")
            gif1.place(x=50 + 500 * (count // 2), y=30 + 300 * (count % 2))
            count += 1
            if count == 4:
                break
            # display GIF
            # for i in range(2):
            #     gif1 = AnimatedGIF(root, save_dir + "/demo" + str(i) + ".gif")
            #     gif1.place(x=20, y=30 + 400*i)
            # for i in range(2):
            #     gif2 = AnimatedGIF(root, save_dir + "/demo" + str(i+5) + ".gif")
            #     gif2.place(x=450, y=30 + 400*i)


    data_button = Button(root, text='data', command=getDataset, width=10)
    data_button.place(x=400, y=600)

    predict_button = Button(root, text='predict', command=predict, width=10)
    predict_button.place(x=500, y=600)

    show_button = Button(root, text='show', command=show, width=10)
    show_button.place(x=600, y=600)

    # l1 = Label(root, text="ground truth", fg="red")
    # l1.place(x=40, y=10)

    # l2 = Label(root, text="our prediction", fg="red")
    # l2.place(x=450, y=10)

    root.mainloop()
