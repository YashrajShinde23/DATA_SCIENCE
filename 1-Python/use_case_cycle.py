#scenerio
#imagine a customer support team where new tickets are assigned
#to available agents in a round robin manner.each agent should
#receive the next ticket in sequence.if the last agent is 
#reached,the cycle should restart from first agent 
#automatically

import itertools

#list of available support agents
agents=["alice","bob","charlie","david"]

#create a cycling iterator over agents
agent_cycle=itertools.cycle(agents)

#simulate incoming support tickets
tickets=["ticket-101","ticket-102","ticket-103","ticket-104","ticket-105","ticket-106"]

#assign tickets to agent in round robin manner
assignments={ticket: next(agent_cycle) for ticket in tickets}

#print the assignment
for ticket,agent in assignments.items():
    print(f"{tickets} -> assigned to: {agent}")





