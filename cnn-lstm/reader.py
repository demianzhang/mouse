import numpy
import os
import codecs as cs
import pickle

def load(data_file = "data.pkl"):
  if os.path.exists(data_file):
    return pickle.load(open(data_file, 'rb'))

  def load_file(fn, has_label=True):
      with cs.open(fn) as src:
          stream = src.read().strip().split('\n')
          points = []
          labels = []
          for line in stream:
              line = line.strip().split(' ')
              sentc = []
              label = []
              line2point=line[1].strip().split(';')
              for point in line2point:
                  if point == line2point[-1]:
                      break
                  point=point.strip().split(',')
                  point[0] = int(point[0])
                  point[1] = int(point[1])
                  point[2] = int(point[2])
                  sentc.append(point)
              if has_label:
                if line[3]=='1':
                    label.append([1.,0.])
                else:
                    label.append([0.,1.])
              else:
                  label.append(None)
              points.append(sentc)
              labels.append(label)
      return points, labels

  train_x, train_y = load_file('training.txt', has_label=True)
  #test_x, test_y = load_file('testing_sample.txt')
  print('train set size: ', len(train_x))
  #print('test set size: ', len(test_x))

  trainX = numpy.asarray(train_x)
  trainY = numpy.asarray(train_y)
  #testX = numpy.asarray(test_x)
  #testY = numpy.asarray(test_y)
  with open(data_file, 'wb') as fout:
      #pickle.dump((trainX, trainY, testX, testY), fout)
      pickle.dump((trainX, trainY), fout)

  return trainX, trainY#, testX, testY

def loadTest():
    def load_file(fn, has_label=False):
        with cs.open(fn) as src:
            stream = src.read().strip().split('\n')
            points = []
            labels = []
            for line in stream:
                line = line.strip().split(' ')
                sentc = []
                label = []
                line2point = line[1].strip().split(';')
                for point in line2point:
                    if point == line2point[-1]:
                        break
                    point = point.strip().split(',')
                    point[0] = int(point[0])
                    point[1] = int(point[1])
                    point[2] = int(point[2])
                    sentc.append(point)
                if has_label:
                    if line[3] == '1':
                        label.append([1., 0.])
                    else:
                        label.append([0., 1.])
                else:
                    label.append(None)
                points.append(sentc)
                labels.append(label)
        return points

    test_x = load_file('test.txt', has_label=False)
    test_x = numpy.asarray(test_x)
    return test_x

if __name__=='__main__':
    X=loadTest()
    print type(X)
    print X[99999]