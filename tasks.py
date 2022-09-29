import json
import random

import machines
import networking

def parse_results(results):
    try:
        parsed = results
        if results is not None:
            if type(results) is list:
                parsed = []
                for e in results:
                    parsed.append(parse_results(e))
            elif type(results) is dict:
                parsed = {}
                for k,v in results.items():
                    parsed[k] = parse_results(v)
            else:
                try:
                    parsed = results.to_dict()
                except:
                    parsed = results
                
        return parsed
    except Exception as e:
        print("Error while parsing results: {}".format(e))
        return {}

def parse_report(r):
    r = parse_results(r)
    try:
        return {
            "success": 100 if r["status"] is True else 0,
            "time": r["time"],
            "iterations": r["iterations"],
            "errors": r["errors"],
            "#errors": {
                "per_type": len(r["errors"]),
                "total": sum([int(r["errors"][e]) for e in r["errors"]])
            },
            "raw": r
        }
    except Exception as e:
        print("Error while parsing report: {}".format(e))
        return {}


def aggregate_reports(ls):
    r = {}
    try:
        errors = {}
        for l in ls:
            if "errors" in l:
                for e in l["errors"]:
                    newE = str(e).split(" ")
                    newE = " ".join(newE[:min(len(newE), 6)])
                    print("Err",e,newE)
                    if newE in errors:
                        errors[newE].append(l["errors"][e])
                    else:
                        errors[newE] = [l["errors"][e]]
        try:
            for e in errors:
                errors[e] = {"total": sum(errors[e]), "avg": sum(errors[e]) / len(ls), "affected": len(errors[e]) / len(ls) * 100}
        except Exception as e:
            print("Error while parsing errors: {}".format(e))
            errors = {}

        r["errors"] = errors

        success_ls = [l for l in ls if l["success"]==100]

        try:
            r["success"] = (len(success_ls) / len(ls))*100
        except Exception as e:
            print("Error while parsing success: {}".format(e))

        try:
            r["time"] = sum([l["time"] for l in success_ls]) / len(success_ls)
        except Exception as e:
            print("Error while parsing time: {}".format(e))

        try:
            r["iterations"] = sum([l["iterations"] for l in success_ls]) / len(success_ls)
        except Exception as e:
            print("Error while parsing iterations: {}".format(e))

        r["#errors"] = {}
        try:
            r["#errors"]["total"] = sum([l["#errors"]["total"] for l in ls]) / len(ls)
        except Exception as e:
            print("Error while parsing #errors total: {}".format(e))

        try:
            r["#errors"]["per_type"] = sum([l["#errors"]["per_type"] for l in ls]) / len(ls)
        except Exception as e:
            print("Error while parsing #errors per_type: {}".format(e))

    except Exception as e:
        print("Error while aggregating reports: {}".format(e))
    return r

def remake(machine, args, *, verbose=True, very_verbose=True):
    results = {}
    metadata = machine.metadata
    if random.random() < args:
        delete = machines.remove_all_machines({"name": machine.name}, [metadata["cloud"]], verbose=very_verbose)[0]
        add = machines.add_machine(machine.name, metadata["type"], metadata["cloud"], metadata, verbose=very_verbose).to_dict()

        try:
            results["delete"] = parse_report(delete)
        except Exception as e:
            print("Error while parsing delete report: {}".format(e))
            results["delete"] = {}

        try:
            results["add"] = {}
            if "create" in add["data"] and add["data"]["create"] != {}:
                results["add"]["create"] = parse_report(add["data"]["create"])
            if "setup" in add["data"] and add["data"]["setup"] != {}:
                results["add"]["setup"] = parse_report(add["data"]["setup"])
                results["add"]["first_access"] = parse_report(add) #first access only if there is need to setup
        except Exception as e:
            print("Error while parsing add report: {}".format(e))
            results["add"] = {}

    return results

def aggregate_remake(ls, *, verbose=True, very_verbose=True):
    add = {
        "create":[],
        "setup":[],
        "first_access":[]
    }

    delete = []
    for l in ls:
        if l != {}:
            if "delete" in l:
                delete.append(l["delete"])
            if "add" in l:
                if "create" in l["add"]:
                    add["create"].append(l["add"]["create"])
                if "setup" in l["add"]: #comment
                    add["setup"].append(l["add"]["setup"]) #tab
                if "first_access" in l["add"]:
                    add["first_access"].append(l["add"]["first_access"])

    return {
        "add": {
            "create": aggregate_reports(add["create"]),
            "setup": aggregate_reports(add["setup"]),
            "first_access": aggregate_reports(add["first_access"])
        },
        "delete": aggregate_reports(delete)
    }


def probe_network(machine, args, *, verbose=True, very_verbose=True):
    return networking.probe_newtork(machines.get_ip(machine))

def aggregate_network(ls, *, verbose=True, very_verbose=True):
    lat_sum = 0
    lat_count = 0
    lat_failed = 0

    upload_sum = 0
    upload_count = 0
    upload_failed = 0

    download_sum = 0
    download_count = 0
    download_failed = 0
    for l in ls:
        if "latency" in l and l["latency"] != None and l["latency"] >= 0:
            lat_sum += l["latency"]
            lat_count += 1
        else:
            lat_failed += 1
        if "bandwidth" in l:
            if "upload" in l["bandwidth"] and l["bandwidth"]["upload"] != None and l["bandwidth"]["upload"] >= 0:
                upload_sum += l["bandwidth"]["upload"]
                upload_count += 1
            else:
                upload_failed += 1
            if "download" in l["bandwidth"] and l["bandwidth"]["download"] != None and l["bandwidth"]["download"] >= 0:
                download_sum += l["bandwidth"]["download"]
                download_count += 1
                download_sum += l["bandwidth"]["download"]
                download_count += 1
            else:
                download_failed += 1
        else:
            upload_failed += 1
            download_failed += 1

    return {
        "latency": {
            "avg": lat_sum / lat_count if lat_count > 0 else None,
            "success": 100 - (lat_failed / len(ls) * 100)
        },
        "bandwidth": {
            "upload": {
                "avg": upload_sum / upload_count if upload_count > 0 else None,
                "success": 100 - (upload_failed / len(ls) * 100)
            },
            "download": {
                "avg": download_sum / download_count if download_count > 0 else None,
                "success": 100 - (download_failed / len(ls) * 100)
            }
        }
    }
                

def exec_script(machine, args, *, verbose=True, very_verbose=True):
    return parse_report(machines.exec_script(machine.name, args, verbose=very_verbose).to_dict())

def aggregate_script(ls, *, verbose=True, very_verbose=True):
    return aggregate_reports(ls)

def fio(machine, args, *, verbose=True, very_verbose=True):
    r = machines.exec_script(machine.name, "fio", {"<SIZE>": args}, verbose=very_verbose).to_dict()
    try:
        d = "\n".join((r["data"].split("\n"))[2:-2])
        return json.loads(d)
    except Exception as e:
        print("Error while parsing fio report: {}".format(e))
        return {}

def average_dicts(ls):
    avg = {}
    head = ls[0]
    for k in head.keys():
        if type(head[k]) is dict:
            avg[k] = average_dicts([l[k] for l in ls])
        elif type(head[k]) is list:
            for i in range(len(head[k])):
                avg[k] = average_dicts([l[k][i] for l in ls])
        elif type(head[k]) is int or type(head[k]) is float:
            avg[k] = sum([l[k] for l in ls]) / len(ls)
        else:
            avg[k] = head[k]

    return avg

def aggregate_fio(ls, *, verbose=True, very_verbose=True):
    new_ls = [l for l in ls if l != {} and l is not None]
    res = {}
    try:
        if len(new_ls) != 0:
            res = average_dicts(new_ls)
        res["success"] = float(len(new_ls) / len(ls)) * 100
    except Exception as e:
        print("Error while aggregating fio report: {}".format(e))
        res = {}
    return res
    