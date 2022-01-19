import matplotlib.pyplot as plt
import argparse
import os
import sys
import glob
import numpy as np

def autoscale_y(ax,margin=0.1):
    """This function rescales the y-axis based on the data that is visible given the current xlim of the axis.
    ax -- a matplotlib axes object
    margin -- the fraction of the total height of the y-data to pad the upper and lower ylims"""

    import numpy as np

    def get_bottom_top(line):
        xd = line.get_xdata()
        yd = line.get_ydata()
        lo,hi = ax.get_xlim()
        y_displayed = yd[((xd>lo) & (xd<hi))]
        y_displayed = [y for y in y_displayed if y != np.inf and y != -np.inf]
        if len(y_displayed) == 0:
            return np.inf, -np.inf
        h = np.max(y_displayed) - np.min(y_displayed)
        bot = np.min(y_displayed)-margin*h
        top = np.max(y_displayed)+margin*h
        return bot,top

    lines = ax.get_lines()
    bot,top = np.inf, -np.inf

    for line in lines:
        new_bot, new_top = get_bottom_top(line)
        if new_bot < bot: bot = new_bot
        if new_top > top: top = new_top

    if bot != np.inf and top != -np.inf:
        ax.set_ylim(bot,top)

def plotWithCheckpointFolder(checkpoint_folder):

    pthList = glob.glob(os.path.join(checkpoint_folder, "*.pth"))
    print(f"Found {len(pthList)} checkpoint files")

    data_x = []
    data_y = []

    for pthFile in pthList:
        split = os.path.basename(pthFile)[:-4].split('-')
        data_x += [int(split[-3])]
        data_y += [float(split[-1])]
    
    if data_x != []:
        print(f'Found best epoch: {data_x[data_y.index(min(data_y))]} with loss of {min(data_y)}')

    # Sort data
    data_x, data_y = zip(*sorted(list(zip(data_x, data_y))))

    if dim == 0:
        axis.plot(data_x, data_y, label="Validation loss",color="blue")
        axis.set(xlabel='Epoch',ylabel='Loss',title=checkpoint_folder, xlim=x_lims, ylim=y_lims)
        if x_lims and not y_lims:
            autoscale_y(axis,margin=0.1)
        axis.legend()        
    elif dim == 1:
        axis[i].plot(data_x, data_y, label="Validation loss",color="blue")
        # axis[i].set(xlabel='Epoch',ylabel='Loss',title=checkpoint_folder)
        axis[i].set(title=checkpoint_folder, xlim=x_lims, ylim=y_lims)
        if x_lims and not y_lims:
            autoscale_y(axis[i],margin=0.1)
        axis[i].legend()
    else:
        axis[i,j].plot(data_x, data_y, label="Validation loss",color="blue")
        # axis[i,j].set(xlabel='Epoch',ylabel='Loss',title=checkpoint_folder)
        axis[i,j].set(title=checkpoint_folder, xlim=x_lims, ylim=y_lims)
        if x_lims and not y_lims:
            autoscale_y(axis[i,j],margin=0.1)
        axis[i,j].legend()

def plotWithLogFile(log_file):

    with open(log_file, 'r') as f:
        lines = f.readlines()

    train_loss_x = []
    train_loss_y = []
    train_regr_loss_x = []
    train_regr_loss_y = []
    train_clas_loss_x = []
    train_clas_loss_y = []
    validation_loss_x = []
    validation_loss_y = []
    validation_regr_loss_x = []
    validation_regr_loss_y = []
    validation_clas_loss_x = []
    validation_clas_loss_y = []

    for l in lines:
        split = [x.strip() for x in l.split(',')]
        epoch = int(split[2].split(':')[-1])

        if split[0] == 'T':
            # training loss
            epoch += eval(split[3].split(':')[1]) - 1
            train_loss_x += [epoch]
            train_loss_y += [float(split[4].split(':')[-1])]

            train_regr_loss_x += [epoch]
            train_regr_loss_y += [float(split[5].split(':')[-1])]

            train_clas_loss_x += [epoch]
            train_clas_loss_y += [float(split[6].split(':')[-1])]

        elif split[0] == 'V':
            # validation loss
            validation_loss_x += [epoch]
            validation_loss_y += [float(split[3].split(':')[-1])]

            validation_regr_loss_x += [epoch]
            validation_regr_loss_y += [float(split[4].split(':')[-1])]

            validation_clas_loss_x += [epoch]
            validation_clas_loss_y += [float(split[5].split(':')[-1])]
        else:
            print(f"Found invalid log line: {l}")

    if validation_loss_x != []:
        print(f'Found best epoch: {validation_loss_x[validation_loss_y.index(min(validation_loss_y))]} with loss of {min(validation_loss_y)}')

    if dim == 0:
        axis.plot(train_loss_x, train_loss_y, label="Training loss",color="red")
        axis.plot(train_regr_loss_x, train_regr_loss_y, label="Train regression loss",color="lightcoral")
        axis.plot(train_clas_loss_x, train_clas_loss_y, label="Train classification loss",color="firebrick")
        axis.plot(validation_loss_x, validation_loss_y, label="Validation loss",color="blue") 
        axis.plot(validation_regr_loss_x, validation_regr_loss_y, label="Validation regression loss",color="lightskyblue")
        axis.plot(validation_clas_loss_x, validation_clas_loss_y, label="Validation classification loss",color="navy")
        axis.set(xlabel='Epoch',ylabel='Loss',title=os.path.normpath(os.path.join(log_file, '..')), xlim=x_lims, ylim=y_lims)
        if x_lims and not y_lims:
            autoscale_y(axis[i],margin=0.1)
        axis.legend()
    elif dim == 1:
        axis[i].plot(train_loss_x, train_loss_y, label="Training loss",color="red")
        axis[i].plot(train_regr_loss_x, train_regr_loss_y, label="Train regression loss",color="lightcoral")
        axis[i].plot(train_clas_loss_x, train_clas_loss_y, label="Train classification loss",color="firebrick")
        axis[i].plot(validation_loss_x, validation_loss_y, label="Validation loss",color="blue") 
        axis[i].plot(validation_regr_loss_x, validation_regr_loss_y, label="Validation regression loss",color="lightskyblue")
        axis[i].plot(validation_clas_loss_x, validation_clas_loss_y, label="Validation classification loss",color="navy")
        # axis[i].set(xlabel='Epoch',ylabel='Loss',title=os.path.normpath(os.path.join(log_file, '..')))
        axis[i].set(title=os.path.normpath(os.path.join(log_file, '..')), xlim=x_lims, ylim=y_lims)
        if x_lims and not y_lims:
            autoscale_y(axis[i],margin=0.1)
        axis[i].legend()
    else:
        axis[i,j].plot(train_loss_x, train_loss_y, label="Training loss",color="red")
        axis[i,j].plot(train_regr_loss_x, train_regr_loss_y, label="Train regression loss",color="lightcoral")
        axis[i,j].plot(train_clas_loss_x, train_clas_loss_y, label="Train classification loss",color="firebrick")
        axis[i,j].plot(validation_loss_x, validation_loss_y, label="Validation loss",color="blue") 
        axis[i,j].plot(validation_regr_loss_x, validation_regr_loss_y, label="Validation regression loss",color="lightskyblue")
        axis[i,j].plot(validation_clas_loss_x, validation_clas_loss_y, label="Validation classification loss",color="navy")
        # axis[i,j].set(xlabel='Epoch',ylabel='Loss',title=os.path.normpath(os.path.join(log_file, '..')))
        axis[i,j].set(title=os.path.normpath(os.path.join(log_file, '..')), xlim=x_lims, ylim=y_lims)
        if x_lims and not y_lims:
            autoscale_y(axis[i,j],margin=0.1)
        axis[i,j].legend()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="This program will plot the validation loss of a pytorch-ssd generated model")
    parser.add_argument("model_dir", metavar="model-dir", nargs='+', help="Model directory")
    parser.add_argument("--x_lims", "-x", metavar="x-lims", nargs='+', help="X limits. Supply one value for lower limit, or two for both limits")
    parser.add_argument("--y_lims", "-y", metavar="x-lims", nargs='+', help="Y limits. Supply one value for lower limit, or two for both limits")
    args = parser.parse_args()


    if args.x_lims:
        if len(args.x_lims) == 1:
            args.x_lims = float(args.x_lims[0])
        elif len(args.x_lims) == 2:
            args.x_lims = [float(x) for x in args.x_lims]
        else:
            print(f"Expected 1 or 2 x limits, got {args.x_lims}")
            exit()

    if args.y_lims:
        if len(args.y_lims) == 1:
            args.y_lims = float(args.y_lims[0])
        elif len(args.y_lims) == 2:
            args.y_lims = [float(y) for y in args.y_lims]
        else:
            print(f"Expected 1 or 2 y limits, got {args.y_lims}")
            exit()

    global x_lims
    global y_lims
    x_lims = args.x_lims
    y_lims = args.y_lims

    n = len(args.model_dir)

    global axis
    global i
    global j
    global dim
    i = 0
    j = 0
    x = 0

    if n >= 16:
        figure, axis = plt.subplots(nrows=int(np.ceil(len(args.model_dir) / 4)), ncols=4, num="Model loss")
    elif n >= 9:
        figure, axis = plt.subplots(nrows=int(np.ceil(len(args.model_dir) / 4)), ncols=4, num="Model loss")
    elif n >= 4:
        figure, axis = plt.subplots(nrows=int(np.ceil(len(args.model_dir) / 2)), ncols=2, num="Model loss")
    else:
        figure, axis = plt.subplots(nrows=len(args.model_dir), ncols=1, num="Model loss")

    if len(np.shape(axis)) == 0:
        # zero-dimensional plot
        dim = 0
        try:
            md = os.path.expanduser(os.path.expandvars(args.model_dir[x]))
        except IndexError:
                axis.axis('off')

        if os.path.exists(os.path.join(md, "log.txt")):
            print(f"Using log file: {os.path.join(md, 'log.txt')}")
            plotWithLogFile(os.path.join(md, "log.txt"))
        elif os.path.exists(md):
            print(f"Using checkpoint files in directory {md}")
            plotWithCheckpointFolder(md)
        else:
            print("Supplied directory does not exist")
            parser.print_help()

        x += 1
    elif len(np.shape(axis)) == 1:
        dim = 1
        # one-dimensional plot
        for i in range(len(axis)):
            try:
                md = os.path.expanduser(os.path.expandvars(args.model_dir[x]))
            except IndexError:
                    axis[i].axis('off')

            if os.path.exists(os.path.join(md, "log.txt")):
                print(f"Using log file: {os.path.join(md, 'log.txt')}")
                plotWithLogFile(os.path.join(md, "log.txt"))
            elif os.path.exists(md):
                print(f"Using checkpoint files in directory {md}")
                plotWithCheckpointFolder(md)
            else:
                print("Supplied directory does not exist")
                parser.print_help()

            x += 1

    else:
        # multi-dimensional plot
        dim = 2
        for i in range(len(axis)):
            for j in range(len(axis[0])):

                try:
                    md = os.path.expanduser(os.path.expandvars(args.model_dir[x]))
                except IndexError:
                        axis[i,j].axis('off')
                        continue

                if os.path.exists(os.path.join(md, "log.txt")):
                    print(f"Using log file: {os.path.join(md, 'log.txt')}")
                    plotWithLogFile(os.path.join(md, "log.txt"))
                elif os.path.exists(md):
                    print(f"Using checkpoint files in directory {md}")
                    plotWithCheckpointFolder(md)
                else:
                    print("Supplied directory does not exist")
                    parser.print_help()

                x += 1

    mngr = plt.get_current_fig_manager()

    # Compatibility with multiple window backends
    b = plt.get_backend()
    if b == 'TkAgg':
        # for 'TkAgg' backend
        mng = plt.get_current_fig_manager()
        mng.resize(*mng.window.maxsize())
    elif b == 'wxAgg':
        # for 'wxAgg' backend
        mng = plt.get_current_fig_manager()
        mng.frame.Maximize(True)
    elif b == 'Qt4Agg':
        # for 'Qt4Agg' backend
        mng = plt.get_current_fig_manager()
        mng.window.showMaximized()
    elif b == 'Qt5Agg':
        # for 'Qt5Agg' backend
        mng = plt.get_current_fig_manager()
        mng.window.showMaximized()

    plt.show()
    

   