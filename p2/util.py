import os
import matplotlib.pyplot as plt
import numpy as np
# these are the fifteen malware classes we're looking for
malware_classes = ["Agent", "AutoRun", "FraudLoad", "FraudPack", "Hupigon", "Krap",
           "Lipler", "Magania", "None", "Poison", "Swizzor", "Tdss",
           "VB", "Virut", "Zbot"]

# a function for writing predictions in the required format
def write_predictions(predictions, ids, outfile):
    """
    assumes len(predictions) == len(ids), and that predictions[i] is the
    index of the predicted class with the malware_classes list above for 
    the executable corresponding to ids[i].
    outfile will be overwritten
    """
    with open(outfile,"w+") as f:
        # write header
        f.write("Id,Prediction\n")
        for i, history_id in enumerate(ids):
            f.write("%s,%d\n" % (history_id, predictions[i]))

def drawProgressBar(percent, barLen = 20):
    progress = ""
    for i in range(barLen):
        if i < int(barLen * percent):
            progress += "="
        elif i == int(barLen * percent):
            progress += ">"
        else:
            progress += " "
    print ("[ %s ] %.2f%%" % (progress, percent * 100))
    #sys.stdout.write("[ %s ] %.2f%%" % (progress, percent * 100) + "\r")

def checkpath(outputpath):
    if not os.path.exists(outputpath):
        os.makedirs(outputpath)
        print "making dir: " + outputpath

def makeplot(x=[0], y=[1], label="ah", xlabel="x", ylabel="y", plotname="test"):
    """make a simple plot"""
    plt.figure(0)
    plt.plot(x, y, 'o' , label=label)
    plt.xlabel(xlabel)
    plt.ylabel("ylabel")
    plt.savefig("Plot/" + plotname + ".png", bbox_inches="tight")
    plt.clf()


def maketypeplot(x=[0], y=[1], label=["ah"], xlabel="x", ylabel="y", plotname="test", title="test"):
    """make a simple plot"""
    print x,y
    plt.figure(0)
    plt.rcdefaults()
    fig, ax = plt.subplots()
    y_pos = np.arange(len(y))
    print y_pos
    ax.bar(y_pos, x, 0.35, align='center', color='green', ecolor='black')
    ax.set_xticks(y_pos)
    ax.set_xticklabels(y, rotation='vertical')
    #ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    plt.savefig("Plot/" + plotname + ".png", bbox_inches="tight")
    plt.clf()