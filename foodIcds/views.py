import base64
from datetime import datetime
from io import BytesIO
from math import pi

# Create your views here.
from django.contrib import messages
from django.http import JsonResponse
from django.shortcuts import render, redirect
from django.template.loader import render_to_string
from django.views.generic.base import View
from pulp import *
from rest_framework import status
import pandas as pd
import numpy as np

# Create your views here
from foodIcds.render import Render

input_EAR = pd.DataFrame()
EAR = pd.DataFrame()
EAR_11 = pd.DataFrame()
obj = 0
final_optimized_cost = 0
final_optimized_cost_lessKcal = 0
opStatus = ''
opStatus_lessKcal = ''


def Percentagecalculation(nutrition, Age_group):
    nutrition = nutrition.sort_index(ascending=False, axis=1)
    EARfull = pd.read_csv("EARFULL.csv", encoding='unicode_escape')
    if Age_group == "child(1-3)yrs":
        input_EAR = EARfull["Gap1_3yrs"]
    if Age_group == "pregnant":
        input_EAR = EARfull["Gap_preg"]
    if Age_group == "lactation":
        input_EAR = EARfull["Gap_lact"]
    if Age_group == "6-12 months":
        input_EAR = EARfull["Gap_6-12-c1"]

    RDAfull = pd.read_csv("RDAFULL.csv", encoding='unicode_escape')
    if Age_group == "child(1-3)yrs":
        input_RDA = RDAfull["Gap1_3yrs"]
    if Age_group == "pregnant":
        input_RDA = RDAfull["Gap_preg"]
    if Age_group == "lactation":
        input_RDA = RDAfull["Gap_lact"]
    if Age_group == "6-12 months":
        input_RDA = RDAfull["Gap_6-12-c1"]

    other_nut = nutrition.drop([3, 4, 5, 6])
    other_units = ["Energy (Kcal)", "Protein (g)", "Fat (g)"]
    other_nut["Nutritions"] = other_units
    units = EARfull["Nutritions"]

    Lab = EARfull["Lab"]
    input_EAR = pd.concat([Lab, input_EAR], axis=1, ignore_index=True)
    input_EAR = input_EAR.rename(columns={0: "Lab", 1: "EAR"})
    input_RDA = pd.concat([Lab, input_RDA], axis=1, ignore_index=True)
    input_RDA = input_RDA.rename(columns={0: "Lab", 1: "RDA"})

    percentage_calculation = nutrition.loc[nutrition['Nutritions'].isin(input_EAR["Lab"])]
    percentage_calculation.index = input_RDA.index
    percentage_calculation["EAR_PERCENT"] = percentage_calculation["Amount"] / input_EAR["EAR"] * 100
    percentage_calculation["RDA_PERCENT"] = percentage_calculation["Amount"] / input_RDA["RDA"] * 100
    percentage_calculation["RDA_PERCENT"] = np.ceil(percentage_calculation["RDA_PERCENT"])
    percentage_calculation["EAR_PERCENT"] = np.ceil(percentage_calculation["EAR_PERCENT"])
    percentage_calculation["Nutritions"] = units

    Fat_percentage = np.ceil(((nutrition["Amount"].iloc[2] * 9) / nutrition["Amount"].iloc[0]) * 100)

    percentage_calculation.columns = ["Nutritions", "Nutrition composition", "Perc. of EAR (%)", "Perc. of RDA (%)"]
    other_nut.columns = ["Nutritions", "Nutrition composition"]
    return percentage_calculation, Fat_percentage, other_nut


def NUTCAL(quantity_food):
    Food = quantity_food["Food_Name"]
    Food = Food.sort_values()

    data = pd.read_csv("Nutrition_sheet_ICDS.csv", encoding='unicode_escape')

    # selecting particular food items based on input function
    data = pd.DataFrame(data.loc[data['Food_Name'].isin(Food)])
    data = data.sort_values(["Food_Name"])

    colnam = ["Food_Group", "Food_Name", "Quantity", "Energy", "Protein", "Fat", "Iron", "Calcium", "Zinc", "Folate",
              "Cost"]

    F = data
    F.columns = colnam
    ##convert to 1g from 100g
    F1 = F.iloc[:, 3:len(F.columns) + 1] / 100
    F = pd.concat([F.iloc[:, 0:2], F1], axis=1)
    F = F.sort_values(["Food_Name"])
    F = F.drop(columns=["Cost", "Food_Name", "Food_Group"])
    ##
    A = F.T

    ##if NA is there make it as zero
    A = A.fillna(0)
    nutritions = pd.DataFrame(A.index)
    ##nutritional_calculation
    ## A and quantity_food

    A1 = np.array(A)
    A2 = np.array(quantity_food["Amount"])

    nut_out = pd.DataFrame(np.dot(A1, A2))

    ou = pd.concat([nut_out, nutritions], axis=1, ignore_index=False)
    ou.columns = ["Amount", "Nutritions"]

    ou = np.round(ou, 1)
    print('ou', ou)
    return ou


def LPPWOVAR_LESSKCAL(Age_group, Food, input_cost, scheme, quantity):
    global input_EAR, EAR, EAR_11, final_optimized_cost_lessKcal
    print(scheme)
    if scheme is None:
        input_cost = input_cost[input_cost["Food_Name"] != "Milk powder"]
    print('input-cost-check', input_cost)

    model = LpProblem("Optimal_diet_plan", LpMinimize)
    input_cost = input_cost.sort_values(['Food_Name'])
    print(input_cost)
    variable_names = input_cost['Food_Name']
    DV_variables = LpVariable.matrix("X", variable_names, cat="Continuous", lowBound=0)

    allocation = np.array(DV_variables)
    print(allocation)
    input_cost = input_cost  # .sort_values(['Food'])
    Cost = input_cost["input_cost"]
    cost_matrix = Cost
    obj_func = lpSum(allocation * cost_matrix)
    print(obj_func)
    model += obj_func
    print(model)

    # input nutrient data and EARfull-----
    data = pd.read_csv("Nutrition_sheet_ICDS.csv", encoding='unicode_escape')
    if scheme is None:
        data = data[data["Food_Name"] != "Milk powder"]
        Food = input_cost['Food_Name']
    # selecting particular food items based on input function
    data = pd.DataFrame(data.loc[data['Food_Name'].isin(Food)])
    Foodgroup = data["Food_Group"]
    data = data.sort_values(['Food_Name'])
    print(data)
    print(Food)

    # input EARRDA2020 DATASET----

    EARfull = pd.read_csv("Nutrition_gap_RDA.csv", encoding='unicode_escape')

    if Age_group == "6-12 months":
        input_EAR = EARfull["Gap_6-12"]

    Lab = EARfull["Lab"]
    input_EAR = pd.concat([input_EAR, Lab], axis=1, ignore_index=True)
    input_EAR = input_EAR.rename(columns={0: "EAR", 1: "Lab"})
    EAR = input_EAR

    if scheme is None:
        milk = pd.read_csv("Milk_powder.csv", encoding='unicode_escape')

        if Age_group == "6-12 months":
            EAR_11 = EAR["EAR"] - milk["Milk_powder"] * (int(quantity) / 100)

        EAR_11[EAR_11 < 0] = 0

        EAR["EAR"] = EAR_11

    F = data
    colnam = ["Food_Group", "Food_Name", "Quantity", "Energy", "Protein", "Fat", "Iron", "Calcium", "Zinc", "Folate",
              "Cost"]
    F.columns = colnam

    # convert to 1g from 100g
    F1 = F.iloc[:, 3:len(F.columns) + 1] / 100
    F = pd.concat([F.iloc[:, 0:2], F1], axis=1)

    A = F.T
    A = A.iloc[2:len(A), :]
    A.columns = F["Food_Name"]

    # min value for each nutrition
    EAR = EAR.loc[EAR["Lab"].isin(F.columns)]
    b = EAR["EAR"]
    b.index = EAR["Lab"]

    A = A.fillna(0)

    # ===================================#
    # Selection----------------

    # nutrient<-c('Energy','Protein','Fat','Fibre','Iron','Calcium','Zinc','Iodine','Magnesium','VB12','VA','VB1','VB2','VB3','VB6','Folate','VC')
    nutrient = ['Energy', 'Protein', "Fat", 'Iron', 'Calcium', 'Zinc', 'Folate']

    mm = pd.DataFrame(A.loc[A.index.isin(nutrient)])
    AA = mm

    for i in range(len(AA)):
        model += lpSum(AA.iloc[i] * allocation) >= b.iloc[i]

    for i in range(len(allocation)):
        model += lpSum(allocation[i]) >= 0

    # constraints for upper limit for protein and energy

    model += lpSum(A.iloc[2,] * allocation) <= lpSum(A.iloc[0,] * allocation) * 0.4 / 9

    if Age_group == "6-12 months":
        Cereals = data[data["Food_Group"] == "Cereals"]
        Cereal = Cereals["Food_Name"]
        Cereal.index = range(len(Cereal))
        print('hello:')
        Pulses = data[data["Food_Group"] == "Pulses"]
        Pulse = Pulses["Food_Name"]
        Pulse.index = range(len(Pulse))

        a = np.zeros((len(AA.columns)))
        for i in range(len(Cereal)):
            a[AA.columns == Cereal[i]] = 1

        for i in range(len(Pulse)):
            a[AA.columns == Pulse[i]] = -2

        model += lpSum(a * allocation) == 0
        print(model)

        # model +=lpSum(Cost*allocation) <=10

    k = data[data["Food_Group"] == "Cereals"]
    fcereals = k["Food_Name"]
    fcereals.index = range(0, len(fcereals), 1)

    k = data[data["Food_Group"] == "Pulses"]
    fpulse = k["Food_Name"]
    fpulse.index = range(0, len(fpulse), 1)
    cereals = [1 / len(fcereals)] * len(fcereals)
    pulse = [1 / len(fpulse)] * len(fpulse)

    if Age_group == "6-12 months":

        # fix minimum limit
        a = np.zeros((len(AA.columns)))
        for i in range(len(fcereals)):
            a[AA.columns == fcereals[i]] = 1
        model += lpSum(a * allocation) >= 20

        a = np.zeros((len(AA.columns)))
        for i in range(len(fpulse)):
            a[AA.columns == fpulse[i]] = 1
        model += lpSum(a * allocation) >= 20

    model += lpSum(allocation) <= 300 / 3 + 300 * 0.20
    # model += lpSum(allocation) >= 300 - 300*0.20

    # Eggs*********************

    if "Egg" in tuple(Foodgroup) and (Age_group == "6-12 months"):
        a = np.zeros((len(AA.columns)))
        a[AA.columns == "Egg"] = 1
        model += lpSum(a * allocation) == 0

    if "Fruits" in tuple(Foodgroup) and (Age_group == "6-12 months"):
        a = np.zeros((len(AA.columns)))
        a[AA.columns == "Banana"] = 1
        model += lpSum(a * allocation) == 0

    if "Oil" in tuple(Foodgroup):
        a = np.zeros((len(AA.columns)))
        a[AA.columns == "Oil"] = 1
        a[AA.columns == "Ghee"] = 1
        model += lpSum(a * allocation) >= 5

    model += lpSum(allocation * cost_matrix) <= 8

    datainput = data[["Food_Name", "Food_Group"]]
    datainput = datainput.sort_values(["Food_Name"])

    sol = model.solve()
    obj = value(model.objective)
    final_optimized_cost_lessKcal = np.round(obj, 2)

    out_food = pd.DataFrame(Food.sort_values())
    k = pd.DataFrame(np.zeros(len(model.variables())))
    j = 0

    for v in model.variables():
        k.loc[j] = v.value()
        j = j + 1

    kk = pd.DataFrame(np.zeros(len(model.variables())))
    j = 0

    for v in model.variables():
        kk.loc[j] = v.name
        j = j + 1

    out1 = pd.concat([k, kk], axis=1, ignore_index=True)

    food_out = out1
    out_food.index = food_out.index

    out2 = pd.concat([out_food, food_out], axis=1, ignore_index=False)
    out2 = pd.DataFrame(out2)
    out2.columns = ["Food_Name", "Amount", "Food_output"]
    final_out_lessKcal = out2[["Food_Name", 'Amount']]
    datainput.index = final_out_lessKcal.index
    final_out_lessKcal = pd.concat([final_out_lessKcal, datainput["Food_Group"]], axis=1)
    final_out_lessKcal["Amount"] = np.ceil(final_out_lessKcal["Amount"])
    global opStatus_lessKcal
    opStatus_lessKcal = LpStatus[model.status]
    print(opStatus_lessKcal)
    input_cost_v1 = input_cost.sort_values(["Food_Name"])
    costperitem = np.array(input_cost_v1["input_cost"])
    quan = np.array(final_out_lessKcal["Amount"])
    c_1 = costperitem * quan
    final_out_lessKcal["cost"] = c_1
    if scheme is None:
        final_out_lessKcal.loc[len(final_out_lessKcal.index)] = ['Milk powder', int(quantity), 'Milk powder', 0]
    print(final_out_lessKcal)
    return final_out_lessKcal


def LPPWOVAR(Age_group, Food, input_cost, scheme, quantity):
    global input_EAR, EAR, EAR_11, final_optimized_cost
    if scheme is None:
        input_cost = input_cost[input_cost["Food_Name"] != "Milk powder"]

    model = LpProblem("Optimal_diet_plan", LpMinimize)
    input_cost = input_cost.sort_values(['Food_Name'])
    print(input_cost)
    variable_names = input_cost['Food_Name']
    DV_variables = LpVariable.matrix("X", variable_names, cat="Continuous", lowBound=0)

    allocation = np.array(DV_variables)
    print(allocation)
    input_cost = input_cost  # .sort_values(['Food'])
    Cost = input_cost["input_cost"]
    cost_matrix = Cost
    obj_func = lpSum(allocation * cost_matrix)
    print(obj_func)
    model += obj_func
    print(model)

    # input nutrient data and EARfull-----
    data = pd.read_csv("Nutrition_sheet_ICDS.csv", encoding='unicode_escape')
    if scheme is None:
        data = data[data["Food_Name"] != "Milk powder"]
        Food = input_cost['Food_Name']
    # selecting particular food items based on input function
    data = pd.DataFrame(data.loc[data['Food_Name'].isin(Food)])
    Foodgroup = data["Food_Group"]
    data = data.sort_values(['Food_Name'])
    print(data)

    # input EARRDA2020 DATASET----

    EARfull = pd.read_csv("Nutrition_gap_RDA.csv", encoding='unicode_escape')
    if Age_group == "child(1-3)yrs":
        input_EAR = EARfull["Gap1_3yrs"]
    if Age_group == "pregnant":
        input_EAR = EARfull["Gap_preg"]
    if Age_group == "lactation":
        input_EAR = EARfull["Gap_lact"]
    if Age_group == "6-12 months":
        input_EAR = EARfull["Gap_6-12-c1"]

    Lab = EARfull["Lab"]
    input_EAR = pd.concat([input_EAR, Lab], axis=1, ignore_index=True)
    input_EAR = input_EAR.rename(columns={0: "EAR", 1: "Lab"})
    EAR = input_EAR
    print('Ear prev', EAR)
    if scheme is None:
        milk = pd.read_csv("Milk_powder.csv", encoding='unicode_escape')
        print('quan', type(quantity))
        EAR_11 = EAR["EAR"] - milk["Milk_powder"] * (quantity / 100)
        print('ear11', EAR_11)
        EAR_11[EAR_11 < 0] = 0
        EAR["EAR"] = EAR_11
    print('Ear is', EAR)
    F = data
    colnam = ["Food_Group", "Food_Name", "Quantity", "Energy", "Protein", "Fat", "Iron", "Calcium", "Zinc", "Folate",
              "Cost"]
    F.columns = colnam

    # convert to 1g from 100g
    F1 = F.iloc[:, 3:len(F.columns) + 1] / 100
    F = pd.concat([F.iloc[:, 0:2], F1], axis=1)

    A = F.T
    A = A.iloc[2:len(A), :]
    A.columns = F["Food_Name"]

    # min value for each nutrition
    EAR = EAR.loc[EAR["Lab"].isin(F.columns)]
    b = EAR["EAR"]
    b.index = EAR["Lab"]

    A = A.fillna(0)

    # ===================================#
    # Selection----------------

    # nutrient<-c('Energy','Protein','Fat','Fibre','Iron','Calcium','Zinc','Iodine','Magnesium','VB12','VA','VB1','VB2','VB3','VB6','Folate','VC')
    nutrient = ['Energy', 'Protein', "Fat", 'Iron', 'Calcium', 'Zinc', 'Folate']

    mm = pd.DataFrame(A.loc[A.index.isin(nutrient)])
    AA = mm

    for i in range(len(AA)):
        model += lpSum(AA.iloc[i] * allocation) >= b.iloc[i]

    for i in range(len(allocation)):
        model += lpSum(allocation[i]) >= 0

    # constraints for upper limit for protein and energy

    model += lpSum(A.iloc[2,] * allocation) <= lpSum(A.iloc[0,] * allocation) * 0.4 / 9

    # cereal
    if (Age_group == "pregnant") or (Age_group == "lactation"):
        Cereals = data[data["Food_Group"] == "Cereals"]
        Cereal = Cereals["Food_Name"]
        Cereal.index = range(len(Cereal))

        Pulses = data[data["Food_Group"] == "Pulses"]
        Pulse = Pulses["Food_Name"]
        Pulse.index = range(len(Pulse))

        a = np.zeros((len(AA.columns)))
        for i in range(len(Cereal)):
            a[AA.columns == Cereal[i]] = 1

        for i in range(len(Pulse)):
            a[AA.columns == Pulse[i]] = -3

        model += lpSum(a * allocation) == 0

    if (Age_group == "child(1-3)yrs") or (Age_group == "6-12 months"):
        Cereals = data[data["Food_Group"] == "Cereals"]
        Cereal = Cereals["Food_Name"]
        Cereal.index = range(len(Cereal))
        print('hello:')
        Pulses = data[data["Food_Group"] == "Pulses"]
        Pulse = Pulses["Food_Name"]
        Pulse.index = range(len(Pulse))

        a = np.zeros((len(AA.columns)))
        for i in range(len(Cereal)):
            a[AA.columns == Cereal[i]] = 1

        for i in range(len(Pulse)):
            a[AA.columns == Pulse[i]] = -2

        model += lpSum(a * allocation) == 0
        print(model)

        # model +=lpSum(Cost*allocation) <=10

    k = data[data["Food_Group"] == "Cereals"]
    fcereals = k["Food_Name"]
    fcereals.index = range(0, len(fcereals), 1)

    k = data[data["Food_Group"] == "Pulses"]
    fpulse = k["Food_Name"]
    fpulse.index = range(0, len(fpulse), 1)
    cereals = [1 / len(fcereals)] * len(fcereals)
    pulse = [1 / len(fpulse)] * len(fpulse)

    # fix minimum limit
    a = np.zeros((len(AA.columns)))
    for i in range(len(fcereals)):
        a[AA.columns == fcereals[i]] = 1
    model += lpSum(a * allocation) >= 30

    a = np.zeros((len(AA.columns)))
    for i in range(len(fpulse)):
        a[AA.columns == fpulse[i]] = 1
    model += lpSum(a * allocation) >= 30

    if (Age_group == "pregnant") or (Age_group == "lactation"):
        model += lpSum(allocation) <= 600 / 3 + 600 * 0.20

    if (Age_group == "6-12 months") or (Age_group == "child(1-3)yrs"):
        model += lpSum(allocation) <= 250
    # model += lpSum(allocation) >= 500 / 3.5

    # Eggs*********************

    if (Age_group == "pregnant") or (Age_group == "lactation"):
        if "Egg" in tuple(Foodgroup) and (Age_group != "child(1-3)yrs"):
            a = np.zeros((len(AA.columns)))
            a[AA.columns == "Egg"] = 1
            model += lpSum(a * allocation) == 45

    if "Oil" in tuple(Foodgroup):
        a = np.zeros((len(AA.columns)))
        a[AA.columns == "Oil"] = 1
        a[AA.columns == "Ghee"] = 1
        model += lpSum(a * allocation) >= 5

    if (Age_group == "child(1-3)yrs") or (Age_group == "6-12 months"):
        model += lpSum(allocation * cost_matrix) <= 8

    if (Age_group == "pregnant") or (Age_group == "lactation"):
        model += lpSum(allocation * cost_matrix) <= 9.5

    datainput = data[["Food_Name", "Food_Group"]]
    datainput = datainput.sort_values(["Food_Name"])

    sol = model.solve()
    obj = value(model.objective)
    print(np.round(obj, 2))
    final_optimized_cost = np.round(obj, 2)

    out_food = pd.DataFrame(Food.sort_values())
    k = pd.DataFrame(np.zeros(len(model.variables())))
    j = 0

    for v in model.variables():
        k.loc[j] = v.value()
        j = j + 1

    kk = pd.DataFrame(np.zeros(len(model.variables())))
    j = 0

    for v in model.variables():
        kk.loc[j] = v.name
        j = j + 1

    out1 = pd.concat([k, kk], axis=1, ignore_index=True)

    food_out = out1
    out_food.index = food_out.index

    out2 = pd.concat([out_food, food_out], axis=1, ignore_index=False)
    out2 = pd.DataFrame(out2)
    out2.columns = ["Food_Name", "Amount", "Food_output"]
    final_out = out2[["Food_Name", 'Amount']]
    datainput.index = final_out.index
    final_out = pd.concat([final_out, datainput["Food_Group"]], axis=1)
    final_out["Amount"] = np.ceil(final_out["Amount"])
    global opStatus
    opStatus = LpStatus[model.status]
    print(opStatus)
    input_cost_v1 = input_cost.sort_values(["Food_Name"])
    costperitem = np.array(input_cost_v1["input_cost"])
    quan = np.array(final_out["Amount"])
    c_1 = costperitem * quan
    final_out["cost"] = c_1
    if scheme is None:
        final_out.loc[len(final_out.index)] = ['Milk powder', quantity, 'Milk powder', 0]
    print(final_out)
    return final_out


class Index(View):

    def get(self, request):
        return render(request, 'icds/index.html')


class Category(View):

    def get(self, request):
        query = request.GET.get('data')
        request.session['query'] = query

        if query == 'THR':
            request.session.pop("infant", None)
            request.session.pop("toddler", None)
            request.session.pop("pregnant", None)
            request.session.pop("lactating", None)
            return render(request, 'icds/category.html')
        return render(request, 'icds/index.html')

    def post(self, request):
        query = request.session['query']
        if query == 'THR':
            context = {
                'data': request.POST,
                'has_error': False
            }
            request.session.pop("infant", None)
            request.session.pop("toddler", None)
            request.session.pop("pregnant", None)
            request.session.pop("lactating", None)

            infant = request.POST.get('infant', None)
            toddler = request.POST.get('toddler', None)
            pregnant = request.POST.get('pregnant', None)
            lactating = request.POST.get('lactating', None)

            if context['has_error']:
                return render(request, 'icds/index.html', context=context,
                              status=status.HTTP_400_BAD_REQUEST)

            request.session['infant'] = int(infant)
            print(infant)
            request.session['toddler'] = int(toddler)
            request.session['pregnant'] = int(pregnant)
            request.session['lactating'] = int(lactating)

            return redirect('FoodSelection')

        return render(request, 'icds/index.html')


class FoodSelection(View):

    def get(self, request):
        if 'query' not in request.session:
            return redirect('home')
        if request.session['query'] == 'THR':
            if 'infant' not in request.session:
                return redirect('home')

            infant = request.session['infant']
            toddler = request.session['toddler']
            pregnant = request.session['pregnant']
            lactating = request.session['lactating']

            if infant <= 0 and toddler <= 0 and pregnant <= 0 and lactating <= 0:
                messages.add_message(request, messages.ERROR, 'Please enter at least one! ')
                return render(request, 'icds/category.html')

            print(infant, toddler, pregnant, lactating)
            return render(request, 'icds/foodSelection.html',
                          {'infant': infant, 'toddler': toddler, 'pregnant': pregnant, 'lactating': lactating})
        return render(request, 'icds/index.html')

    def post(self, request):
        if 'query' not in request.session:
            return redirect('home')

        if request.session['query'] == 'THR':
            if 'infant' not in request.session:
                messages.add_message(request, messages.ERROR, 'Please enter the number of people again! ')
                return redirect('category')

            Cereals = request.POST.getlist('Cereals', None)
            Pulses = request.POST.getlist('Pulses', None)
            Others = request.POST.getlist('Others', None)
            milkpowder = request.POST.getlist('milkpowder', None)
            print('MILK-POWDER', milkpowder)
            scheme = request.POST.get('scheme', None)
            milkPowderQuantity = request.POST.get('milkPowderQuantity', None)

            toddlersCereals = request.POST.getlist('toddlersCereals', None)
            toddlersPulses = request.POST.getlist('toddlersPulses', None)
            toddlersOthers = request.POST.getlist('toddlersOthers', None)
            toddlersmilkpowder = request.POST.getlist('toddlersmilkpowder', None)
            toddlersscheme = request.POST.get('toddlersscheme', None)
            toddlersmilkPowderQuantity = request.POST.get('toddlersmilkPowderQuantity', None)

            pregnantCereals = request.POST.getlist('pregnantCereals', None)
            pregnantPulses = request.POST.getlist('pregnantPulses', None)
            pregnantOthers = request.POST.getlist('pregnantOthers', None)
            pregnantmilkpowder = request.POST.getlist('pregnantmilkpowder', None)
            pregnantscheme = request.POST.get('pregnantscheme', None)
            pregnantmilkPowderQuantity = request.POST.get('pregnantmilkPowderQuantity', None)

            lactatingCereals = request.POST.getlist('lactatingCereals', None)
            lactatingPulses = request.POST.getlist('lactatingPulses', None)
            lactatingOthers = request.POST.getlist('lactatingOthers', None)
            lactatingmilkpowder = request.POST.getlist('lactatingmilkpowder', None)
            lactatingscheme = request.POST.get('lactatingscheme', None)
            lactatingmilkPowderQuantity = request.POST.get('lactatingmilkPowderQuantity', None)

            request.session['Cereals'] = Cereals
            request.session['Pulses'] = Pulses
            request.session['Others'] = Others
            request.session['toddlersCereals'] = toddlersCereals
            request.session['toddlersPulses'] = toddlersPulses
            request.session['toddlersOthers'] = toddlersOthers
            request.session['pregnantCereals'] = pregnantCereals
            request.session['pregnantPulses'] = pregnantPulses
            request.session['pregnantOthers'] = pregnantOthers
            request.session['lactatingCereals'] = lactatingCereals
            request.session['lactatingPulses'] = lactatingPulses
            request.session['lactatingOthers'] = lactatingOthers
            request.session['milkpowder'] = milkpowder
            request.session['scheme'] = scheme
            request.session['milkPowderQuantity'] = milkPowderQuantity

            request.session['toddlersmilkpowder'] = toddlersmilkpowder
            request.session['toddlersscheme'] = toddlersscheme
            request.session['toddlersmilkPowderQuantity'] = toddlersmilkPowderQuantity

            request.session['pregnantmilkpowder'] = pregnantmilkpowder
            request.session['pregnantscheme'] = pregnantscheme
            request.session['pregnantmilkPowderQuantity'] = pregnantmilkPowderQuantity

            request.session['lactatingmilkpowder'] = lactatingmilkpowder
            request.session['lactatingscheme'] = lactatingscheme
            request.session['lactatingmilkPowderQuantity'] = lactatingmilkPowderQuantity

            return redirect('FoodCost')
        return render(request, 'icds/index.html')


class FoodCost(View):

    def get(self, request):
        if 'query' not in request.session:
            return redirect('home')
        if request.session['query'] == 'THR':
            if 'infant' not in request.session:
                # messages.add_message(request, messages.ERROR, 'Please enter the number of people again! ')
                return redirect('home')

            infant = request.session['infant']
            toddler = request.session['toddler']
            pregnant = request.session['pregnant']
            lactating = request.session['lactating']

            Cereals = request.session['Cereals']
            Pulses = request.session['Pulses']
            Others = request.session['Others']
            milkpowder = request.session['milkpowder']
            infantFood = itertools.chain(Cereals, Pulses, Others, milkpowder)
            infantFood = list(infantFood)
            request.session['infantFood'] = infantFood

            toddlersCereals = request.session['toddlersCereals']
            toddlersPulses = request.session['toddlersPulses']
            toddlersOthers = request.session['toddlersOthers']
            toddlersmilkpowder = request.session['toddlersmilkpowder']
            toddlersFood = itertools.chain(toddlersCereals, toddlersPulses, toddlersOthers, toddlersmilkpowder)
            toddlersFood = list(toddlersFood)
            request.session['toddlersFood'] = toddlersFood

            pregnantCereals = request.session['pregnantCereals']
            pregnantPulses = request.session['pregnantPulses']
            pregnantOthers = request.session['pregnantOthers']
            pregnantmilkpowder = request.session['pregnantmilkpowder']
            pregnantFood = itertools.chain(pregnantCereals, pregnantPulses, pregnantOthers, pregnantmilkpowder)
            pregnantFood = list(pregnantFood)
            request.session['pregnantFood'] = pregnantFood

            lactatingCereals = request.session['lactatingCereals']
            lactatingPulses = request.session['lactatingPulses']
            lactatingOthers = request.session['lactatingOthers']
            lactatingmilkpowder = request.session['lactatingmilkpowder']
            lactatingFood = itertools.chain(lactatingCereals, lactatingPulses, lactatingOthers, lactatingmilkpowder)
            lactatingFood = list(lactatingFood)
            request.session['lactatingFood'] = lactatingFood
            data = pd.read_csv('Nutrition_sheet_ICDS.csv', encoding='unicode_escape')
            data = data[['Food_Name', 'Cost']]
            data['Cost'] = data['Cost'] * 10

            food_cost = {}
            for index, row in data.iterrows():
                food_cost[row['Food_Name']] = row['Cost']

            # resultCereals = list(set(Cereals + toddlersCereals + pregnantCereals + lactatingCereals))
            resultCereals = sorted(np.unique(Cereals + toddlersCereals + pregnantCereals + lactatingCereals))
            print(resultCereals)
            resultPulses = sorted(np.unique(Pulses + toddlersPulses + pregnantPulses + lactatingPulses))
            print(resultPulses)
            resultOthers = sorted(np.unique(Others + toddlersOthers + pregnantOthers + lactatingOthers))
            print(resultOthers)
            resultMilkPowder = sorted(
                np.unique(milkpowder + toddlersmilkpowder + pregnantmilkpowder + lactatingmilkpowder))
            print(resultMilkPowder)

            cereal_cost = []
            pulse_cost = []
            other_cost = []
            milk_cost = []

            for cereal in resultCereals:
                if cereal in food_cost:
                    print(cereal, food_cost[cereal])
                    cereal_cost.append(food_cost[cereal])

            for pulse in resultPulses:
                if pulse in food_cost:
                    print(pulse, food_cost[pulse])
                    pulse_cost.append(food_cost[pulse])

            for other in resultOthers:
                if other in food_cost:
                    print(other, food_cost[other])
                    other_cost.append(food_cost[other])

            for milk in resultMilkPowder:
                if milk in food_cost:
                    print(milk, food_cost[milk])
                    milk_cost.append(food_cost[milk])

            return render(request, 'icds/foodCostAll.html',
                          {'infant': infant, 'toddler': toddler, 'pregnant': pregnant, 'lactating': lactating,
                           'infantFood': infantFood, 'toddlersFood': toddlersFood, 'pregnantFood': pregnantFood,
                           'lactatingFood': lactatingFood,
                           'resultCereals': resultCereals, 'resultPulses': resultPulses, 'resultOthers': resultOthers,
                           'resultMilkPowder': resultMilkPowder,
                           'cereal_cost': cereal_cost, 'pulse_cost': pulse_cost, 'other_cost': other_cost,
                           'milk_cost': milk_cost})

        return render(request, 'icds/index.html')

    def post(self, request):
        Jowar = request.POST.get('Jowar', None)
        Ragi = request.POST.get('Ragi', None)
        Wheat = request.POST.get('Wheat', None)
        WheatflourAtta = request.POST.get('Wheat flour Atta', None)
        WheatBroken = request.POST.get('Wheat Broken', None)
        Maize = request.POST.get('Maize', None)
        Rava = request.POST.get('Rava', None)
        Rice = request.POST.get('Rice', None)

        RajmaRed = request.POST.get('Rajma Red (Kidney beans)', None)
        SoyabeanBrown = request.POST.get('Soya bean (Brown)', None)
        SoyabeanWhite = request.POST.get('Soya bean (White)', None)
        Bengalgram = request.POST.get('Bengal Gram (Channa dal)', None)
        Blackgram = request.POST.get('Black Gram (Urad dal)', None)
        YellowLentil = request.POST.get('Yellow Lentil (Mung dal)', None)
        Peas = request.POST.get('Peas', None)
        RedGram = request.POST.get('Red Gram (Arhar dal)(Toor dal)', None)
        GreenGram = request.POST.get('Green Gram (Moong dal)', None)
        RedLentil = request.POST.get('Red Lentil (Masoor dal)', None)

        Jaggery = request.POST.get('Jaggery', None)
        Sugar = request.POST.get('Sugar', None)
        Ghee = request.POST.get('Ghee', None)
        Oil = request.POST.get('Oil', None)
        Egg = request.POST.get('Egg', None)
        Banana = request.POST.get('Banana', None)
        Groundnut = request.POST.get('Ground nut', None)
        MilkPowder = request.POST.get('Milk powder', None)

        cost_list = dict(
            {'Jowar': Jowar, 'Ragi': Ragi, 'Wheat': Wheat, 'Wheat flour Atta': WheatflourAtta,
             'Wheat Broken': WheatBroken, 'Maize': Maize, 'Rava': Rava, 'Rice': Rice,
             'Rajma Red (Kidney beans)': RajmaRed, 'Soya bean (Brown)': SoyabeanBrown,
             'Soya bean (White)': SoyabeanWhite, 'Bengal Gram (Channa dal)': Bengalgram,
             'Black Gram (Urad dal)': Blackgram, 'Yellow Lentil (Mung dal)': YellowLentil, 'Peas': Peas,
             'Red Gram (Arhar dal)(Toor dal)': RedGram, 'Green Gram (Moong dal)': GreenGram,
             'Red Lentil (Masoor dal)': RedLentil,
             'Jaggery': Jaggery, 'Sugar': Sugar, 'Ghee': Ghee, 'Oil': Oil, 'Egg': Egg, 'Banana': Banana,
             'Ground nut': Groundnut, 'Milk powder': MilkPowder
             })
        cost_list = dict(filter(lambda item: item[1] is not None, cost_list.items()))

        request.session['cost_list'] = cost_list
        print(cost_list)
        return redirect('result')


class Result(View):

    def get(self, request):
        if ('query' not in request.session) and ('infant' not in request.session):
            return redirect('home')

        infant = request.session['infant']
        toddler = request.session['toddler']
        pregnant = request.session['pregnant']
        lactating = request.session['lactating']
        Cereals = request.session['Cereals']
        Pulses = request.session['Pulses']
        Others = request.session['Others']
        scheme = request.session['scheme']
        milkPowderQuantity = request.session['milkPowderQuantity']
        if milkPowderQuantity is not None:
            milkPowderQuantity = int(milkPowderQuantity)
        cereal_prop = []
        pulse_prop = []
        other_prop = []
        final_q = ''
        if request.session['infant'] > 0:
            cost_list = request.session['cost_list']
            for k, v in cost_list.items():
                cost_list[k] = float(v)

            cost_list = pd.DataFrame(cost_list.items(), columns=['Food_Name', 'input_cost'])
            print(cost_list)

            infantFood = request.session['infantFood']
            print(infantFood)
            infantFoodCost = cost_list[cost_list['Food_Name'].isin(infantFood)]
            print(infantFoodCost)

            infantFoodCost['input_cost'] = infantFoodCost['input_cost'] / 1000
            Age_group = "6-12 months"
            Food = infantFoodCost['Food_Name']

            final_out_infant = LPPWOVAR(Age_group, Food, infantFoodCost, scheme, milkPowderQuantity)

            final_out_infant['Amount'] = np.ceil(final_out_infant['Amount'])
            final_prop_infant = final_out_infant.set_index('Food_Name')['Amount'].to_dict()
            print(final_prop_infant)

            for cereal in Cereals:
                if cereal in final_prop_infant:
                    print(cereal, final_prop_infant[cereal])
                    cereal_prop.append(final_prop_infant[cereal])

            for pulse in Pulses:
                if pulse in final_prop_infant:
                    print(pulse, final_prop_infant[pulse])
                    pulse_prop.append(final_prop_infant[pulse])

            for other in Others:
                if other in final_prop_infant:
                    print(other, final_prop_infant[other])
                    other_prop.append(final_prop_infant[other])

            print(final_optimized_cost)
            nutrition_calc = NUTCAL(final_out_infant)
            final_q = nutrition_calc.set_index('Nutritions')['Amount'].to_dict()
            print(final_q)

        return render(request, 'icds/result.html',
                      {'infant': infant, 'toddler': toddler, 'pregnant': pregnant, 'lactating': lactating,
                       'Cereals': Cereals, 'Pulses': Pulses, 'Others': Others, 'cereal_prop': cereal_prop,
                       'pulse_prop': pulse_prop, 'other_prop': other_prop, 'final_q': final_q,
                       'final_optimized_cost': final_optimized_cost, 'opStatus': opStatus
                       })


def filter_data(request):
    category_id = request.GET.get("category_id")
    print(category_id)

    if category_id == '6 mo to 1 yr':
        Cereals = request.session['Cereals']
        Pulses = request.session['Pulses']
        Others = request.session['Others']
        milkpowder = request.session['milkpowder']
        scheme = request.session['scheme']
        milkPowderQuantity = request.session['milkPowderQuantity']
        if milkPowderQuantity is not None:
            milkPowderQuantity = int(milkPowderQuantity)
        cereal_prop = []
        pulse_prop = []
        other_prop = []
        milk_prop = []
        cereal_prop_lessKcal = []
        pulse_prop_lessKcal = []
        other_prop_lessKcal = []
        milk_prop_lessKcal = []

        if request.session['infant'] > 0:
            cost_list = request.session['cost_list']
            for k, v in cost_list.items():
                cost_list[k] = float(v)

            cost_list = pd.DataFrame(cost_list.items(), columns=['Food_Name', 'input_cost'])
            print(cost_list)

            infantFood = request.session['infantFood']
            print(infantFood)
            infantFoodCost = cost_list[cost_list['Food_Name'].isin(infantFood)]
            print(infantFoodCost)

            infantFoodCost['input_cost'] = infantFoodCost['input_cost'] / 1000
            Age_group = "6-12 months"
            Food = infantFoodCost['Food_Name']

            final_out_infant = LPPWOVAR(Age_group, Food, infantFoodCost, scheme, milkPowderQuantity)

            final_out_infant['Amount'] = np.ceil(final_out_infant['Amount'])
            final_prop_infant = final_out_infant.set_index('Food_Name')['Amount'].to_dict()
            print(final_prop_infant)

            for cereal in Cereals:
                if cereal in final_prop_infant:
                    print(cereal, final_prop_infant[cereal])
                    cereal_prop.append(final_prop_infant[cereal])

            request.session['cereal_prop'] = cereal_prop

            for pulse in Pulses:
                if pulse in final_prop_infant:
                    print(pulse, final_prop_infant[pulse])
                    pulse_prop.append(final_prop_infant[pulse])

            request.session['pulse_prop'] = pulse_prop

            for other in Others:
                if other in final_prop_infant:
                    print(other, final_prop_infant[other])
                    other_prop.append(final_prop_infant[other])

            request.session['other_prop'] = other_prop

            for milk in milkpowder:
                if milk in final_prop_infant:
                    print(milk, final_prop_infant[milk])
                    milk_prop.append(final_prop_infant[milk])

            request.session['milk_prop'] = milk_prop

            print(final_optimized_cost)
            nutrition_calc = NUTCAL(final_out_infant)
            final_q = nutrition_calc.set_index('Nutritions')['Amount'].to_dict()
            print(final_q)

            # less calories intake calculation

            final_out_infant_lessKcal = LPPWOVAR_LESSKCAL(Age_group, Food, infantFoodCost, scheme, milkPowderQuantity)
            final_out_infant_lessKcal['Amount'] = np.ceil(final_out_infant_lessKcal['Amount'])
            final_prop_infant_lessKcal = final_out_infant_lessKcal.set_index('Food_Name')['Amount'].to_dict()
            print(final_prop_infant_lessKcal)

            for cereal in Cereals:
                if cereal in final_prop_infant_lessKcal:
                    print(cereal, final_prop_infant_lessKcal[cereal])
                    cereal_prop_lessKcal.append(final_prop_infant_lessKcal[cereal])

            request.session['cereal_prop_lessKcal'] = cereal_prop_lessKcal

            for pulse in Pulses:
                if pulse in final_prop_infant_lessKcal:
                    print(pulse, final_prop_infant_lessKcal[pulse])
                    pulse_prop_lessKcal.append(final_prop_infant_lessKcal[pulse])

            request.session['pulse_prop_lessKcal'] = pulse_prop_lessKcal

            for other in Others:
                if other in final_prop_infant_lessKcal:
                    print(other, final_prop_infant_lessKcal[other])
                    other_prop_lessKcal.append(final_prop_infant_lessKcal[other])

            request.session['other_prop_lessKcal'] = other_prop_lessKcal

            for milk in milkpowder:
                if milk in final_prop_infant_lessKcal:
                    print(milk, final_prop_infant_lessKcal[milk])
                    milk_prop_lessKcal.append(final_prop_infant_lessKcal[milk])

            request.session['milk_prop_lessKcal'] = milk_prop_lessKcal

            nutrition_calc_lessKcal = NUTCAL(final_out_infant_lessKcal)
            final_q_less = nutrition_calc_lessKcal.set_index('Nutritions')['Amount'].to_dict()

            i = render_to_string('icds/resultInfant.html',
                                 {
                                     'Cereals': Cereals, 'Pulses': Pulses, 'Others': Others, 'milkpowder': milkpowder,
                                     'cereal_prop': cereal_prop, 'pulse_prop': pulse_prop, 'other_prop': other_prop,
                                     'milk_prop': milk_prop,
                                     'final_q': final_q, 'final_optimized_cost': final_optimized_cost,
                                     'opStatus': opStatus,
                                     'cereal_prop_lessKcal': cereal_prop_lessKcal,
                                     'pulse_prop_lessKcal': pulse_prop_lessKcal,
                                     'other_prop_lessKcal': other_prop_lessKcal,
                                     'milk_prop_lessKcal': milk_prop_lessKcal,
                                     'final_q_less': final_q_less,
                                     'final_optimized_cost_lessKcal': final_optimized_cost_lessKcal,
                                     'opStatus_lessKcal': opStatus_lessKcal, 'scheme': scheme
                                 })
            return JsonResponse({'data': i})

    if category_id == '1 yr to 3 yrs':
        toddlersCereals = request.session['toddlersCereals']
        toddlersPulses = request.session['toddlersPulses']
        toddlersOthers = request.session['toddlersOthers']
        toddlersmilkpowder = request.session['toddlersmilkpowder']
        toddlersscheme = request.session['toddlersscheme']
        toddlersmilkPowderQuantity = request.session['toddlersmilkPowderQuantity']
        if toddlersmilkPowderQuantity is not None:
            toddlersmilkPowderQuantity = int(toddlersmilkPowderQuantity)
        cereal_prop_toddler = []
        pulse_prop_toddler = []
        other_prop_toddler = []
        milk_prop_toddler = []
        if request.session['toddler'] > 0:
            print('toddler here ---')
            cost_list = request.session['cost_list']
            for k, v in cost_list.items():
                cost_list[k] = float(v)

            cost_list = pd.DataFrame(cost_list.items(), columns=['Food_Name', 'input_cost'])
            print(cost_list)

            toddlersFood = request.session['toddlersFood']
            print(toddlersFood)
            toddlersFoodCost = cost_list[cost_list['Food_Name'].isin(toddlersFood)]
            print(toddlersFoodCost)

            toddlersFoodCost['input_cost'] = toddlersFoodCost['input_cost'] / 1000
            Age_group = "child(1-3)yrs"
            Food = toddlersFoodCost['Food_Name']

            final_out_toddler = LPPWOVAR(Age_group, Food, toddlersFoodCost, toddlersscheme, toddlersmilkPowderQuantity)

            final_out_toddler['Amount'] = np.ceil(final_out_toddler['Amount'])
            final_prop_toddler = final_out_toddler.set_index('Food_Name')['Amount'].to_dict()
            print(final_prop_toddler)

            for cereal in toddlersCereals:
                if cereal in final_prop_toddler:
                    print(cereal, final_prop_toddler[cereal])
                    cereal_prop_toddler.append(final_prop_toddler[cereal])

            request.session['cereal_prop_toddler'] = cereal_prop_toddler

            for pulse in toddlersPulses:
                if pulse in final_prop_toddler:
                    print(pulse, final_prop_toddler[pulse])
                    pulse_prop_toddler.append(final_prop_toddler[pulse])

            request.session['pulse_prop_toddler'] = pulse_prop_toddler

            for other in toddlersOthers:
                if other in final_prop_toddler:
                    print(other, final_prop_toddler[other])
                    other_prop_toddler.append(final_prop_toddler[other])

            request.session['other_prop_toddler'] = other_prop_toddler

            for milk in toddlersmilkpowder:
                if milk in final_prop_toddler:
                    print(milk, final_prop_toddler[milk])
                    milk_prop_toddler.append(final_prop_toddler[milk])

            request.session['milk_prop_toddler'] = milk_prop_toddler

            print(final_optimized_cost)
            nutrition_calc = NUTCAL(final_out_toddler)
            final_q = nutrition_calc.set_index('Nutritions')['Amount'].to_dict()
            print(final_q)
            t = render_to_string('icds/resultToddler.html',
                                 {
                                     'toddlersCereals': toddlersCereals, 'toddlersPulses': toddlersPulses,
                                     'toddlersOthers': toddlersOthers, 'cereal_prop_toddler': cereal_prop_toddler,
                                     'pulse_prop_toddler': pulse_prop_toddler, 'other_prop_toddler': other_prop_toddler,
                                     'milk_prop_toddler': milk_prop_toddler,
                                     'final_q': final_q, 'final_optimized_cost': final_optimized_cost,
                                     'toddlersscheme': toddlersscheme,
                                     'opStatus': opStatus
                                 })
            return JsonResponse({'data': t})

    if category_id == 'pregnant women':
        pregnantCereals = request.session['pregnantCereals']
        pregnantPulses = request.session['pregnantPulses']
        pregnantOthers = request.session['pregnantOthers']
        pregnantmilkpowder = request.session['pregnantmilkpowder']
        pregnantscheme = request.session['pregnantscheme']
        pregnantmilkPowderQuantity = request.session['pregnantmilkPowderQuantity']
        if pregnantmilkPowderQuantity is not None:
            pregnantmilkPowderQuantity = int(pregnantmilkPowderQuantity)
        cereal_prop_pregnant = []
        pulse_prop_pregnant = []
        other_prop_pregnant = []
        milk_prop_pregnant = []

        if request.session['pregnant'] > 0:
            cost_list = request.session['cost_list']
            for k, v in cost_list.items():
                cost_list[k] = float(v)

            cost_list = pd.DataFrame(cost_list.items(), columns=['Food_Name', 'input_cost'])
            print(cost_list)

            pregnantFood = request.session['pregnantFood']
            print(pregnantFood)
            pregnantFoodCost = cost_list[cost_list['Food_Name'].isin(pregnantFood)]
            print(pregnantFoodCost)

            pregnantFoodCost['input_cost'] = pregnantFoodCost['input_cost'] / 1000
            Age_group = "pregnant"
            Food = pregnantFoodCost['Food_Name']

            final_out_pregnant = LPPWOVAR(Age_group, Food, pregnantFoodCost, pregnantscheme, pregnantmilkPowderQuantity)

            final_out_pregnant['Amount'] = np.ceil(final_out_pregnant['Amount'])
            final_prop_pregnant = final_out_pregnant.set_index('Food_Name')['Amount'].to_dict()
            print(final_prop_pregnant)

            for cereal in pregnantCereals:
                if cereal in final_prop_pregnant:
                    print(cereal, final_prop_pregnant[cereal])
                    cereal_prop_pregnant.append(final_prop_pregnant[cereal])

            for pulse in pregnantPulses:
                if pulse in final_prop_pregnant:
                    print(pulse, final_prop_pregnant[pulse])
                    pulse_prop_pregnant.append(final_prop_pregnant[pulse])

            for other in pregnantOthers:
                if other in final_prop_pregnant:
                    print(other, final_prop_pregnant[other])
                    other_prop_pregnant.append(final_prop_pregnant[other])

            for milk in pregnantmilkpowder:
                if milk in final_prop_pregnant:
                    print(milk, final_prop_pregnant[milk])
                    milk_prop_pregnant.append(final_prop_pregnant[milk])

            print(final_optimized_cost)
            nutrition_calc = NUTCAL(final_out_pregnant)
            final_q = nutrition_calc.set_index('Nutritions')['Amount'].to_dict()
            print(final_q)
            p = render_to_string('icds/resultPregnant.html',
                                 {
                                     'pregnantCereals': pregnantCereals, 'pregnantPulses': pregnantPulses,
                                     'pregnantOthers': pregnantOthers, 'cereal_prop_pregnant': cereal_prop_pregnant,
                                     'pulse_prop_pregnant': pulse_prop_pregnant,
                                     'other_prop_pregnant': other_prop_pregnant,
                                     'milk_prop_pregnant': milk_prop_pregnant,
                                     'final_q': final_q, 'final_optimized_cost': final_optimized_cost,
                                     'pregnantscheme': pregnantscheme,
                                     'opStatus': opStatus
                                 })
            return JsonResponse({'data': p})

    if category_id == 'lactating women':
        lactatingCereals = request.session['lactatingCereals']
        lactatingPulses = request.session['lactatingPulses']
        lactatingOthers = request.session['lactatingOthers']
        lactatingmilkpowder = request.session['lactatingmilkpowder']
        lactatingscheme = request.session['lactatingscheme']
        lactatingmilkPowderQuantity = request.session['lactatingmilkPowderQuantity']
        if lactatingmilkPowderQuantity is not None:
            lactatingmilkPowderQuantity = int(lactatingmilkPowderQuantity)
        cereal_prop_lactating = []
        pulse_prop_lactating = []
        other_prop_lactating = []
        milk_prop_lactating = []
        if request.session['lactating'] > 0:

            cost_list = request.session['cost_list']
            for k, v in cost_list.items():
                cost_list[k] = float(v)

            cost_list = pd.DataFrame(cost_list.items(), columns=['Food_Name', 'input_cost'])
            print(cost_list)

            lactatingFood = request.session['lactatingFood']
            print(lactatingFood)
            lactatingFoodCost = cost_list[cost_list['Food_Name'].isin(lactatingFood)]
            print(lactatingFoodCost)

            lactatingFoodCost['input_cost'] = lactatingFoodCost['input_cost'] / 1000
            Age_group = "lactation"
            Food = lactatingFoodCost['Food_Name']

            final_out_lactating = LPPWOVAR(Age_group, Food, lactatingFoodCost, lactatingscheme,
                                           lactatingmilkPowderQuantity)

            final_out_lactating['Amount'] = np.ceil(final_out_lactating['Amount'])
            final_prop_lactating = final_out_lactating.set_index('Food_Name')['Amount'].to_dict()
            print(final_prop_lactating)

            for cereal in lactatingCereals:
                if cereal in final_prop_lactating:
                    print(cereal, final_prop_lactating[cereal])
                    cereal_prop_lactating.append(final_prop_lactating[cereal])

            for pulse in lactatingPulses:
                if pulse in final_prop_lactating:
                    print(pulse, final_prop_lactating[pulse])
                    pulse_prop_lactating.append(final_prop_lactating[pulse])

            for other in lactatingOthers:
                if other in final_prop_lactating:
                    print(other, final_prop_lactating[other])
                    other_prop_lactating.append(final_prop_lactating[other])

            for milk in lactatingmilkpowder:
                if milk in final_prop_lactating:
                    print(milk, final_prop_lactating[milk])
                    milk_prop_lactating.append(final_prop_lactating[milk])

            print(final_optimized_cost)
            nutrition_calc = NUTCAL(final_out_lactating)
            final_q = nutrition_calc.set_index('Nutritions')['Amount'].to_dict()

            print(final_q, opStatus)

            lw = render_to_string('icds/resultLactating.html',
                                  {
                                      'lactatingCereals': lactatingCereals, 'lactatingPulses': lactatingPulses,
                                      'lactatingOthers': lactatingOthers,
                                      'cereal_prop_lactating': cereal_prop_lactating,
                                      'pulse_prop_lactating': pulse_prop_lactating,
                                      'other_prop_lactating': other_prop_lactating,
                                      'milk_prop_lactating': milk_prop_lactating,
                                      'final_q': final_q, 'final_optimized_cost': final_optimized_cost,
                                      'lactatingscheme': lactatingscheme,
                                      'opStatus': opStatus
                                  })
            return JsonResponse({'data': lw})


# getting pdf for result
class GetPdf(View):
    def get(self, request):
        if ('query' not in request.session) and ('infant' not in request.session):
            return redirect('home')
        infant = request.session['infant']
        toddler = request.session['toddler']
        pregnant = request.session['pregnant']
        lactating = request.session['lactating']

        lact_data = ''
        lact_total = 0
        lact_fat_perc = 0
        lact_perc = pd.DataFrame()
        lact_other_nut = pd.DataFrame()

        preg_data = ''
        preg_total = 0
        preg_perc = pd.DataFrame()
        preg_other_nut = pd.DataFrame()
        preg_fat_perc = 0

        todd_data = ''
        todd_total = 0
        todd_perc = pd.DataFrame()
        todd_other_nut = pd.DataFrame()
        todd_fat_perc = 0

        inft_data = ''
        inft_total = 0
        inft_perc = pd.DataFrame()
        inft_fat_perc = pd.DataFrame()
        inft_other_nut = 0

        inft_data_lessKcal = ''
        inft_total_lessKcal = 0
        inft_perc_lessKcal = pd.DataFrame()
        inft_other_nut_lessKcal = pd.DataFrame()
        inft_fat_perc_lessKcal = 0

        cost_list = request.session['cost_list']
        for k, v in cost_list.items():
            cost_list[k] = float(v)

        cost_list = pd.DataFrame(cost_list.items(), columns=['Food_Name', 'input_cost'])
        print(cost_list)

        if infant > 0:
            scheme = request.session['scheme']
            milkPowderQuantity = request.session['milkPowderQuantity']
            if milkPowderQuantity is not None:
                milkPowderQuantity = int(milkPowderQuantity)
            infantFood = request.session['infantFood']
            print(infantFood)
            infantFoodCost = cost_list[cost_list['Food_Name'].isin(infantFood)]
            print(infantFoodCost)

            infantFoodCost['input_cost'] = infantFoodCost['input_cost'] / 1000
            Age_group = "6-12 months"
            Food = infantFoodCost['Food_Name']

            final_out_infant = LPPWOVAR(Age_group, Food, infantFoodCost, scheme, milkPowderQuantity)
            inft_nutrition = NUTCAL(final_out_infant)

            final_out_infant = final_out_infant[final_out_infant['Amount'] > 0]
            final_out_infant.columns = ['Food Name', 'Per Person Intake (gm)', 'Food Group', 'Per Person Cost (Rs)']
            final_out_infant.reset_index(inplace=True, drop=True)
            final_out_infant = final_out_infant.drop(['Food Group'], axis=1)
            final_out_infant['Total Quantity (gm)'] = final_out_infant['Per Person Intake (gm)'] * lactating
            final_out_infant['Total Cost (Rs)'] = final_out_infant['Per Person Cost (Rs)'] * lactating
            inft_data = final_out_infant.to_html(classes='mystyle')
            inft_total = final_out_infant['Total Cost (Rs)'].sum()
            inft_total = inft_total.round(2)

            inft_perc, inft_fat_perc, inft_other_nut = Percentagecalculation(inft_nutrition, Age_group)
            inft_perc = inft_perc.to_html(classes='mystyle')
            inft_other_nut = inft_other_nut.to_html(classes='mystyle')

            # 6 month to 1 year 250Kcal
            final_out_infant_lessKcal = LPPWOVAR_LESSKCAL(Age_group, Food, infantFoodCost, scheme, milkPowderQuantity)
            inft_nutrition_lessKcal = NUTCAL(final_out_infant_lessKcal)

            final_out_infant_lessKcal = final_out_infant_lessKcal[final_out_infant_lessKcal['Amount'] > 0]
            final_out_infant_lessKcal.columns = ['Food Name', 'Per Person Intake (gm)', 'Food Group',
                                                 'Per Person Cost (Rs)']
            final_out_infant_lessKcal.reset_index(inplace=True, drop=True)
            final_out_infant_lessKcal = final_out_infant_lessKcal.drop(['Food Group'], axis=1)
            final_out_infant_lessKcal['Total Quantity (gm)'] = final_out_infant_lessKcal[
                                                                   'Per Person Intake (gm)'] * lactating
            final_out_infant_lessKcal['Total Cost (Rs)'] = final_out_infant_lessKcal['Per Person Cost (Rs)'] * lactating
            inft_data_lessKcal = final_out_infant_lessKcal.to_html(classes='mystyle')

            inft_total_lessKcal = final_out_infant_lessKcal['Total Cost (Rs)'].sum()
            inft_total_lessKcal = inft_total_lessKcal.round(2)

            inft_perc_lessKcal, inft_fat_perc_lessKcal, inft_other_nut_lessKcal = Percentagecalculation(
                inft_nutrition_lessKcal, Age_group)
            inft_perc_lessKcal = inft_perc_lessKcal.to_html(classes='mystyle')
            inft_other_nut_lessKcal = inft_other_nut_lessKcal.to_html(classes='mystyle')

        if toddler > 0:
            toddlersscheme = request.session['toddlersscheme']
            toddlersmilkPowderQuantity = request.session['toddlersmilkPowderQuantity']
            if toddlersmilkPowderQuantity is not None:
                toddlersmilkPowderQuantity = int(toddlersmilkPowderQuantity)
            toddlersFood = request.session['toddlersFood']
            print(toddlersFood)
            toddlersFoodCost = cost_list[cost_list['Food_Name'].isin(toddlersFood)]
            print(toddlersFoodCost)

            toddlersFoodCost['input_cost'] = toddlersFoodCost['input_cost'] / 1000
            Age_group = "child(1-3)yrs"
            Food = toddlersFoodCost['Food_Name']

            final_out_toddler = LPPWOVAR(Age_group, Food, toddlersFoodCost, toddlersscheme, toddlersmilkPowderQuantity)
            todd_nutrition = NUTCAL(final_out_toddler)

            final_out_toddler = final_out_toddler[final_out_toddler['Amount'] > 0]
            final_out_toddler.columns = ['Food Name', 'Per Person Intake (gm)', 'Food Group', 'Per Person Cost (Rs)']
            final_out_toddler.reset_index(inplace=True, drop=True)
            final_out_toddler = final_out_toddler.drop(['Food Group'], axis=1)
            final_out_toddler['Total Quantity (gm)'] = final_out_toddler['Per Person Intake (gm)'] * lactating
            final_out_toddler['Total Cost (Rs)'] = final_out_toddler['Per Person Cost (Rs)'] * lactating
            todd_data = final_out_toddler.to_html(classes='mystyle')

            todd_total = final_out_toddler['Total Cost (Rs)'].sum()
            todd_total = todd_total.round(2)

            todd_perc, todd_fat_perc, todd_other_nut = Percentagecalculation(todd_nutrition, Age_group)
            todd_perc = todd_perc.to_html(classes='mystyle')
            todd_other_nut = todd_other_nut.to_html(classes='mystyle')

        if pregnant > 0:
            pregnantscheme = request.session['pregnantscheme']
            pregnantmilkPowderQuantity = request.session['pregnantmilkPowderQuantity']
            if pregnantmilkPowderQuantity is not None:
                pregnantmilkPowderQuantity = int(pregnantmilkPowderQuantity)
            pregnantFood = request.session['pregnantFood']
            print(pregnantFood)
            pregnantFoodCost = cost_list[cost_list['Food_Name'].isin(pregnantFood)]
            print(pregnantFoodCost)

            pregnantFoodCost['input_cost'] = pregnantFoodCost['input_cost'] / 1000
            Age_group = "pregnant"
            Food = pregnantFoodCost['Food_Name']

            final_out_pregnant = LPPWOVAR(Age_group, Food, pregnantFoodCost, pregnantscheme, pregnantmilkPowderQuantity)
            preg_nutrition = NUTCAL(final_out_pregnant)

            final_out_pregnant = final_out_pregnant[final_out_pregnant['Amount'] > 0]
            final_out_pregnant.columns = ['Food Name', 'Per Person Intake (gm)', 'Food Group', 'Per Person Cost (Rs)']
            final_out_pregnant.reset_index(inplace=True, drop=True)
            final_out_pregnant = final_out_pregnant.drop(['Food Group'], axis=1)
            final_out_pregnant['Total Quantity (gm)'] = final_out_pregnant['Per Person Intake (gm)'] * lactating
            final_out_pregnant['Total Cost (Rs)'] = final_out_pregnant['Per Person Cost (Rs)'] * lactating
            preg_data = final_out_pregnant.to_html(classes='mystyle')

            preg_total = final_out_pregnant['Total Cost (Rs)'].sum()
            preg_total = preg_total.round(2)

            preg_perc, preg_fat_perc, preg_other_nut = Percentagecalculation(preg_nutrition, Age_group)
            preg_perc = preg_perc.to_html(classes='mystyle')
            preg_other_nut = preg_other_nut.to_html(classes='mystyle')

        if lactating > 0:
            lactatingscheme = request.session['lactatingscheme']
            lactatingmilkPowderQuantity = request.session['lactatingmilkPowderQuantity']
            if lactatingmilkPowderQuantity is not None:
                lactatingmilkPowderQuantity = int(lactatingmilkPowderQuantity)
            lactatingFood = request.session['lactatingFood']
            print(lactatingFood)
            lactatingFoodCost = cost_list[cost_list['Food_Name'].isin(lactatingFood)]
            print(lactatingFoodCost)

            lactatingFoodCost['input_cost'] = lactatingFoodCost['input_cost'] / 1000
            Age_group = "lactation"
            Food = lactatingFoodCost['Food_Name']

            final_out_lactating = LPPWOVAR(Age_group, Food, lactatingFoodCost, lactatingscheme,
                                           lactatingmilkPowderQuantity)
            lact_nutrition = NUTCAL(final_out_lactating)
            final_out_lactating = final_out_lactating[final_out_lactating['Amount'] > 0]
            final_out_lactating.columns = ['Food Name', 'Per Person Intake (gm)', 'Food Group', 'Per Person Cost (Rs)']
            final_out_lactating.reset_index(inplace=True, drop=True)
            final_out_lactating = final_out_lactating.drop(['Food Group'], axis=1)
            final_out_lactating['Total Quantity (gm)'] = final_out_lactating['Per Person Intake (gm)'] * lactating
            final_out_lactating['Total Cost (Rs)'] = final_out_lactating['Per Person Cost (Rs)'] * lactating
            lact_data = final_out_lactating.to_html(classes='mystyle')

            lact_total = final_out_lactating['Total Cost (Rs)'].sum()
            lact_total = lact_total.round(2)
            lact_perc, lact_fat_perc, lact_other_nut = Percentagecalculation(lact_nutrition, Age_group)
            lact_perc = lact_perc.to_html(classes='mystyle')
            lact_other_nut = lact_other_nut.to_html(classes='mystyle')

        params = {
            'today': datetime.now(),
            'infant': infant, 'toddler': toddler, 'pregnant': pregnant, 'lactating': lactating,
            'todd_data': todd_data, 'inft_data': inft_data, 'lact_data': lact_data,
            'preg_data': preg_data, 'inft_data_lessKcal': inft_data_lessKcal,
            'inft_total': inft_total, 'inft_total_lessKcal': inft_total_lessKcal, 'todd_total': todd_total,
            'preg_total': preg_total,
            'lact_total': lact_total,
            'lact_perc': lact_perc, 'lact_other_nut': lact_other_nut, 'lact_fat_perc': lact_fat_perc,
            'preg_perc': preg_perc, 'preg_fat_perc': preg_fat_perc, 'preg_other_nut': preg_other_nut,
            'todd_perc': todd_perc, 'todd_fat_perc': todd_fat_perc, 'todd_other_nut': todd_other_nut,
            'inft_perc_lessKcal': inft_perc_lessKcal, 'inft_fat_perc_lessKcal': inft_fat_perc_lessKcal,
            'inft_other_nut_lessKcal': inft_other_nut_lessKcal,
            'inft_perc': inft_perc, 'inft_fat_perc': inft_fat_perc, 'inft_other_nut': inft_other_nut
        }
        return Render.render('icds/pdf.html', params)
