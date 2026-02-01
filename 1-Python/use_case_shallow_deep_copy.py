import copy

report_template={
    "name":"",
    "scores":[0,0,0],
    "remarks":"pending"
    }

#shallow copy for a new student
student1_report=copy.copy(report_template)
student1_report["name"]="alice"
student1_report["scores"][0]=85

print(report_template["scores"])#output:[85,0,0]

#########################################
import copy

report_template={
    "name":"",
    "scores":[0,0,0],
    "remarks":"pending"
    }

#deep copy for complete isolation
student2_report=copy.deepcopy(report_template)
student2_report["name"]="bob"
student2_report["scores"][0]=92

print(report_template["scores"])


###################################
import copy

params={
        "layers":[64,128,256],
        "activation":"relu"
        }

experiment1=copy.deepcopy(params)
experiment2=copy.deepcopy(params)

experiment1["layers"][0]=32

print(params["layers"])#no change in base params






#use case of filter
users=[
       {"name":"alice","role":"admin"},
       {"name":"bob","role":"editor"},
       {"name":"charlie","role":"admin"}
       ]
admins=list(filter(lambda user: user["role"]== "admin",users))
print(admins)


