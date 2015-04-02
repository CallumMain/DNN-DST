#! /usr/bin/python
import sys,os,argparse,shutil,glob,json,copy,time,operator
from collections import defaultdict
from collections import namedtuple
from argparse import RawTextHelpFormatter
from pprint import pprint

SV_Hypo = namedtuple("SV_Hypo", ["slot", "value"])

def what_slot(mact) :
    # get context for "this" in inform(=dontcare)
    this_slot = None
    
    for act in mact :
        if act["act"] == "request" :
            this_slot = act["slots"][0][1]
        elif act["act"] == "select" :
            this_slot = act["slots"][0][0]
        elif act["act"] == "expl-conf" :
            this_slot = act["slots"][0][0]
    
    return this_slot

def pause() :
    raw_input('press enter to continue ...')

def Uacts(turn) :
    # return merged slu-hyps, replacing "this" with the correct slot
    mact = []
    if "dialog-acts" in turn["output"] :
        mact = turn["output"]["dialog-acts"]
    this_slot = None
    for act in mact :
        if act["act"] == "request" :
            this_slot = act["slots"][0][1]
    this_output = []
    for slu_hyp in turn['input']["live"]['slu-hyps'] :
        score = slu_hyp['score']
        this_slu_hyp = slu_hyp['slu-hyp']
        these_hyps =  []
        for  hyp in this_slu_hyp :
            for i in range(len(hyp["slots"])) :
                slot,value = hyp["slots"][i]
                if slot == "this" :
                    hyp["slots"][i][0] = this_slot
                elif slot == None and value == "dontcare" :
                    hyp["slots"][i][0] = what_slot(mact)
            these_hyps.append(hyp)
        this_output.append((score, these_hyps))
    this_output.sort(key=lambda x:x[0], reverse=True)
    return this_output

def get_slot_value(dact) :
    a = dact["act"]
    s = None;
    v = None;
    if dact["slots"] :
        if len(dact["slots"][0]) >= 1 :
            s = dact["slots"][0][0]
        if len(dact["slots"][0]) == 2 :
            v = dact["slots"][0][1]
    elif dact["act"] :
        s = "act"
        v = dact["act"]
    return a,s,v

def get_slot_value_sys(act) :
    #for system actions
    a = act["act"]
    sv = act["slots"]
    if len(sv) == 1 :
    	s = sv[0][0]
        v = sv[0][1]
    elif len(sv) > 1 :
        s = []
        v = []
        for i in range(len(sv)) :
            s.append(sv[i][0])
            v.append(sv[i][1])
    else :
        s = None
        v = None
    return a,s,v

def remove_dup(mylist) :
    #remove duplicated acts
    newlist = []
    for elm in mylist :
        if not elm in newlist :
            newlist.append(elm)
    return newlist

def get_sv_hyps(slu_hyps,mact) :
    #get slot-value hypotheses
    hyps = []
    for score, uact in slu_hyps :
        uact = remove_dup(uact)
        for dact in uact :
            a,s,v = get_slot_value(dact)
            if s != None and s != 'slot' and s != 'act' and v != None :
                hyps.append(SV_Hypo(slot=s,value=v))
    for act in mact :
        a,s,v = get_slot_value_sys(act)
        if a in ['expl-conf', 'impl-conf', 'inform', 'offer', 'select'] :
            if s != None and s != 'slot' and s != 'act' and v != None :
                if isinstance(s, list) :
                    for i in range(len(s)) :
                        hyps.append(SV_Hypo(slot=s[i],value=v[i]))
                else:
                    hyps.append(SV_Hypo(slot=s,value=v))

    hyps = remove_dup(hyps)
    return hyps

def MachineActKeys():
    keys = ["hello", "bye", "goback" "restart", "null", "ack", "hold-on", "open-request", "bebrief", "sorry", "please-repeat", "please-rephrase", "are-you-there", "didnthear", "impl-conf", "expl-conf", "request", "schedule", "morebuses", "canthelp.nonextbus", "canthelp.route_doesnt_run", "cantjelp.no_connection", "canthelp.uncovered_route", "canthelp.uncovered_stop", "canthelp.cant_find_stop", "canthelp.no_buses_at_time", "canthelp.from_equals_to"]
    
    return keys

def UserActKeys():
    keys = ["hello", "bye", "goback", "restart", "null", "repeat", "nextbus", "prevbus", "tellchoices", "affirm", "negate", "deny", "inform"]

    return keys

def MachineActs(feat):
    keys = ["hello", "bye", "goback" "restart", "null", "ack", "hold-on", "open-request", "bebrief", "sorry", "please-repeat", "please-rephrase", "are-you-there", "didnthear", "impl-conf", "expl-conf", "request", "schedule", "morebuses", "canthelp.nonextbus", "canthelp.route_doesnt_run", "cantjelp.no_connection", "canthelp.uncovered_route", "canthelp.uncovered_stop", "canthelp.cant_find_stop", "canthelp.no_buses_at_time", "canthelp.from_equals_to"]

    for i in xrange(0,len(keys)):
        feat[keys[i]] = 0

def UserActs(feat):
    keys = ["hello", "bye", "goback", "restart", "null", "repeat", "nextbus", "prevbus", "tellchoices", "affirm", "negate", "deny", "inform"]
    
    for i in xrange(0,len(keys)):
        feat[keys[i]] = 0

def init_features() :
    feat = dict();
    feat["SLU-score"] = 0
    feat["Rank"] = 0
    feat["Affirm"] = 0
    feat["Negate"] = 0
    feat["GoBack"] = 0
    feat["Implicit"] = 0
    feat["CantHelp"] = 0
    feat["SlotConfirmed"] = 0
    feat["SlotRequested"] = 0
    feat["SlotInformed"] = 0
    MachineActs(feat)
    UserActs(feat)

    return feat

def update_feat_turn_sys(feat,mact) :
    #system actions
    for act in mact :
        if act["act"] in MachineActKeys() :
            feat[act["act"]] = 1
    return feat


def update_feat_turn_slu(feat,slu_hyps) :
    #user acts in the current turn that don't have slot-value operands
    for score, uact in slu_hyps :
        uact = remove_dup(uact)
        for dact in uact :
            a,s,v = get_slot_value(dact)
            if a in UserActKeys() :
                feat[a] = feat[a] + score;
    return feat

def update_feat_slu_score(feat,slu_hyps, hyp) :
    for score, uact in slu_hyps :
        uact = remove_dup(uact)
        for dact in uact :
            a,s,v = get_slot_value(dact)
            if (s == hyp.slot and v == hyp.value):
                feat["SLU-score"] = score
    return feat

def update_feat_self_rank(hyps_feat,hyp) :
    #rank in the n-best list
    feat = hyps_feat[hyp.slot][hyp.value]
    rank = 1
    for (v,f) in hyps_feat[hyp.slot].items() :
        if v == hyp.value : continue
        if f["SLU-score"] > feat["SLU-score"] :
            rank = rank + 1
    feat["Rank"] = 1/float(rank)
    return feat

def update_feat_user_acts(feat, slu_hyps, hyp, mact) :
    for score, uact in slu_hyps :
        uact = remove_dup(uact)
        for dact in uact :
            a,s,v = get_slot_value(dact)
            #for mact in hist :
            for act in mact :
                am,sm,vm = get_slot_value_sys(act)
                if hyp.slot == sm and hyp.value == vm :
                     if (am == "impl-conf" or am == "expl-conf") and a == "affirm":
                         feat["Affirm"] = score
                     if (am == "impl-conf" or am == "expl-conf") and a == "negate":
                         feat["Negate"] = score
                     if am == "goback":
                         feat["GoBack"] = score
                     if (am == "impl-conf") :
                         feat["Implicit"] = score

    return feat

def update_feat_sys_acts(feat, hyp, mact):
    for act in mact:
        a,s,v = get_slot_value_sys(act)
        if a == "schedule":
            feat["SlotInformed"] = 1
        if a == "request":
            if isinstance(s, list) :
                for i in range(len(s)) :
                    if s[i] in hyp.slot:
                        feat["SlotRequested"] = 1
            elif s in hyp.slot:
                feat["SlotRequested"] = 1
        if s == hyp.slot and v == hyp.value :
            if a in ["canthelp.nonextbus", "canthelp.route_doesnt_run", "cantjelp.no_connection", "canthelp.uncovered_route", "canthelp.uncovered_stop", "canthelp.cant_find_stop", "canthelp.no_buses_at_time", "canthelp.from_equals_to"]:
                feat["CantHelp"] = 1
            if a in ["impl-conf", "expl-conf"]:
                feat["SlotConfirmed"] = 1

    return feat

class vectorizer(object):
    def __init__(self):
        self.reset()
    
    def addTurn(self, turn, labels):
        if "dialog-acts" in turn["output"] :
            mact = turn["output"]["dialog-acts"]
        else :
            mact = []
        
	goal_labels = {}
	if labels and "slu-labels" in labels.keys():
	    for lab in labels["slu-labels"] :
	        if lab["label"] :
		    for s,v in lab["slots"].items():
		        goal_labels[s] = v
        
        slu_hyps = Uacts(turn)
        self.sv_hyps.extend(get_sv_hyps(slu_hyps,mact))
        feat = init_features()
        
        feat = update_feat_turn_sys(feat,mact)
        feat = update_feat_turn_slu(feat,slu_hyps)
        
        hyps_feat = dict()
        for h in self.sv_hyps :
            if h.slot not in hyps_feat.keys() :
                hyps_feat[h.slot] = dict()
            if h.value not in hyps_feat[h.slot].keys() :
                hyps_feat[h.slot][h.value] = copy.deepcopy(feat)
        
        for h in self.sv_hyps :
            hyps_feat[h.slot][h.value] = update_feat_slu_score(hyps_feat[h.slot][h.value],slu_hyps,h)
        for h in self.sv_hyps :
            hyps_feat[h.slot][h.value] = update_feat_self_rank(hyps_feat,h)
        for h in self.sv_hyps :
            hyps_feat[h.slot][h.value] = update_feat_user_acts(hyps_feat[h.slot][h.value],slu_hyps,h,mact)
        for h in self.sv_hyps :
            hyps_feat[h.slot][h.value] = update_feat_sys_acts(hyps_feat[h.slot][h.value],h,mact)
        
        
        self.hist.append(mact)

        for k in hyps_feat.keys():
	    if k == 'act':
		del hyps_feat[k]
            for j in hyps_feat[k].keys():
                if j == 'null':
                    del hyps_feat[k][j]
    

        data_point = dict()
        data_point["x"] = hyps_feat
	if labels and "slu-labels" in labels.keys():
	     data_point["y"] = goal_labels
        return data_point
    
    def reset(self):
        self.hist = []
        self.sv_hyps = []


def clip(x) :
    if x > 1:
        return 1
    if x < 0:
        return 0
    return x


def is_zero(x) :
    epsilon = 1e-9
    return math.fabs(x) < epsilon


def main(argv):
    #
    # CMD LINE ARGS
    # 
    install_path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    utils_dirname = os.path.join(install_path,'lib')
    sys.path.append(utils_dirname)
    from dataset_walker import dataset_walker
    list_dir = os.path.join(install_path,'config')

    parser = argparse.ArgumentParser(description='Simple hand-crafted dialog state tracker baseline.')
    parser.add_argument('--dataset', dest='dataset', action='store', metavar='DATASET', required=True,
                        help='The dataset to analyze, for example train1 or test2 or train3a')
    parser.add_argument('--dataroot',dest='dataroot',action='store',required=True,metavar='PATH',
                        help='Will look for corpus in <destroot>/<dataset>/...')
    parser.add_argument('--datafile',dest='datafile',action='store',required=True,metavar='JSON_FILE',
                            help='File to write output')
    parser.add_argument('--label',dest='label',action='store',required=True,metavar='BOOL',
                            help='load labels')
    
    args = parser.parse_args()
    
    label = False
    if args.label.lower() == 'true':
	label = True

    dataset = dataset_walker(args.dataset,dataroot=args.dataroot,labels=label)
    
    datafile = open(args.datafile, "wb")
    data = {"sessions":[]}
    data["dataset"] = args.dataset
    
    vector = vectorizer()
    
    for call in dataset :
        this_session = {"session-id":call.log["session-id"], "turns":[]}
        vector.reset()
        for turn, labels in call :
            data_point = vector.addTurn(turn,labels)
            this_session["turns"].append(data_point)
        
        data["sessions"].append(this_session)
    
    json.dump(data, datafile,indent=4)

    datafile.close()

if (__name__ == '__main__'):
    main(sys.argv)
