"""
@author: Caglar Aytekin
contact: caglar@deepcause.ai 
"""
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score as accuracy
from sklearn.metrics import roc_auc_score
from torch.optim.lr_scheduler import StepLR
import numpy as np 
class Trainer:
    def __init__(self, model, X_train, X_val, y_train, y_val,lr,batch_size,epochs,problem_type):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.problem_type=problem_type
        if self.problem_type==0:
            self.criterion = nn.MSELoss()
        elif self.problem_type==1:
            self.criterion = nn.BCEWithLogitsLoss()
        elif self.problem_type==2:
            self.criterion = nn.CrossEntropyLoss()
            y_train=y_train.squeeze().long()
            y_val=y_val.squeeze().long()
            
            
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        self.train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(dataset=val_dataset, batch_size=len(val_dataset), shuffle=False)
        self.batch_size=batch_size
        self.epochs=epochs
        self.best_metric = float('inf') if problem_type == 0 else float('-inf')
        self.scheduler = StepLR(self.optimizer, step_size=epochs//3, gamma=0.2)

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        total=0
        correct=0
        for inputs, labels in self.train_loader:
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            total += len(labels.squeeze())
            if self.problem_type==1:
                correct += (torch.round(torch.sigmoid(outputs.data)).squeeze() == labels.squeeze()).sum().item()
            elif self.problem_type==2:
                correct += (torch.max(outputs.data, 1)[1] == labels.squeeze()).sum().item()
        return total_loss/len(self.train_loader) , correct/total

    def validate(self):
        self.model.eval()
        val_loss = 0
        total=0
        val_predictions = []
        val_targets = []
        with torch.no_grad():
            for inputs, labels in self.val_loader:
                outputs = self.model(inputs)
                val_loss += self.criterion(outputs, labels).item()
                total += len(labels.squeeze())
                if self.problem_type==1:
                    val_predictions.extend(torch.sigmoid(outputs).view(-1).cpu().numpy())
                elif self.problem_type==2:
                    val_predictions.extend(torch.max(outputs.data, 1)[1].view(-1).cpu().numpy())
                val_targets.extend(labels.view(-1).cpu().numpy())

            if self.problem_type==1:
                val_roc_auc =roc_auc_score(val_targets, val_predictions)
                val_acc = accuracy(val_targets, np.round(val_predictions))
            elif self.problem_type==2:
                val_acc = accuracy(val_targets,val_predictions)
                val_roc_auc=0
            else:
                val_roc_auc=0
                val_acc=0
        return val_loss /len(self.val_loader), val_acc,val_roc_auc

    def train(self):
        for epoch in range(self.epochs):
            #Increase alpha up to 1-tenth of entire epochs
            alpha_now=np.minimum(1.0,float(epoch)/float(self.epochs/10))
            # print(alpha_now)
            self.model.set_alpha(alpha_now)
            if epoch>self.epochs//10:
                save_permit=True
            else:
                save_permit=False
            tr_loss, tr_acc = self.train_epoch()
            val_loss, val_acc , val_roc_auc= self.validate()
            
            if self.problem_type == 0:
                print(f'Epoch {epoch}: Train Loss {tr_loss:.4f}, Val Loss {val_loss:.4f}')
                if (val_loss < self.best_metric)and(save_permit):
                    self.best_metric = val_loss
                    # Save model checkpoint
                    self.model.nninput=None #Delete data remaining from training 
                    self.encodings=None
                    self.taus=None
                    torch.save(self.model, 'best_model.pth')
                    # print("Saving model with best validation loss.")
            
            # Problem type 1: Focus on loss, accuracy, and AUC
            elif self.problem_type == 1:
                print(f'Epoch {epoch}: Train Loss {tr_loss:.4f}, Train Acc {tr_acc:.4f}, Val Loss {val_loss:.4f}, Val Acc {val_acc:.4f}, Val ROC AUC {val_roc_auc:.4f}')
                if (val_roc_auc > self.best_metric)and(save_permit):
                    self.best_metric = val_roc_auc
                    # Save model checkpoint
                    self.model.nninput=None #Delete data remaining from training 
                    self.encodings=None
                    self.taus=None
                    torch.save(self.model, 'best_model.pth')
                    # print("Saving model with best validation ROC AUC.")
            
            # Problem type 2: Focus on loss and accuracy
            elif self.problem_type == 2:
                print(f'Epoch {epoch}: Train Loss {tr_loss:.4f}, Train Acc {tr_acc:.4f}, Val Loss {val_loss:.4f}, Val Acc {val_acc:.4f}')
                if (val_acc > self.best_metric)and(save_permit):
                    self.best_metric = val_acc
                    # Save model checkpoint
                    self.model.nninput=None #Delete data remaining from training 
                    self.encodings=None
                    self.taus=None
                    torch.save(self.model, 'best_model.pth')
                    # print("Saving model with best validation accuracy.")
            self.scheduler.step()
        # Load best validation model
        # self.model.load_state_dict(torch.load('best_model.pth'))

        self.model = torch.load('best_model.pth')

        
        
    def evaluate(self,X_test, y_test):
        test_loader=DataLoader(dataset=TensorDataset(X_test, y_test), batch_size=len(y_test), shuffle=True)
        self.model.eval()
        test_loss = 0
        total=0
        test_predictions = []
        test_targets = []
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = self.model(inputs)
                test_loss += self.criterion(outputs, labels).item()
                total += len(labels.squeeze())
                if self.problem_type==1:
                    test_predictions.extend(torch.sigmoid(outputs).view(-1).cpu().numpy())
                elif self.problem_type==2:
                    test_predictions.extend(torch.max(outputs.data, 1)[1].view(-1).cpu().numpy())
                test_targets.extend(labels.view(-1).cpu().numpy())

            if self.problem_type==1:
                test_roc_auc =roc_auc_score(test_targets, test_predictions)
                test_acc = accuracy(test_targets, np.round(test_predictions))
                print('ROC-AUC: ', test_roc_auc)
                return test_roc_auc
            elif self.problem_type==2:
                test_acc = accuracy(test_targets,test_predictions)
                test_roc_auc=0
                print('ACC: ', test_acc)
                return test_acc
            else:
                test_roc_auc=0
                test_acc=0
                print('MSE: ', test_loss /len(test_loader))
                return test_loss /len(test_loader)
            
   
