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
from foodIcds.render import Render

# Create your views here

input_EAR = pd.DataFrame()
EAR = pd.DataFrame()
EAR_11 = pd.DataFrame()
obj = 0
final_optimized_cost = 0
final_optimized_cost_lessKcal = 0
opStatus = ''
opStatus_lessKcal = ''


def PercentagecalculationHCM(nutrition, Age_group):
    nutrition = nutrition.sort_index(ascending=False, axis=1)
    EARfull = pd.read_csv("EARFULL_HCM.csv", encoding='unicode_escape')
    if Age_group == "child(4-6)yrs":
        input_EAR = EARfull["Gap4_6yrs"]
    if Age_group == "pregnant":
        input_EAR = EARfull["Gap_preg"]
    if Age_group == "lactation":
        input_EAR = EARfull["Gap_lact"]

    RDAfull = pd.read_csv("RDAFULL_HCM.csv", encoding='unicode_escape')
    if Age_group == "child(4-6)yrs":
        input_RDA = RDAfull["Gap4_6yrs"]
    if Age_group == "pregnant":
        input_RDA = RDAfull["Gap_preg"]
    if Age_group == "lactation":
        input_RDA = RDAfull["Gap_lact"]

    other_nut = nutrition.drop([3, 4, 5, 6])
    other_units = ["Energy (Kcal)", "Protein (g)", "Fat (g)"]
    other_nut["Nutritions"] = other_units
    units = EARfull["Nutritions"]

    Lab = EARfull["Lab"]
    input_EAR = pd.concat([input_EAR, Lab], axis=1, ignore_index=True)
    input_EAR = input_EAR.rename(columns={0: "EAR", 1: "Lab"})
    input_RDA = pd.concat([input_RDA, Lab], axis=1, ignore_index=True)
    input_RDA = input_RDA.rename(columns={0: "RDA", 1: "Lab"})

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


def LPPWOVARHCM(Age_group, Food, input_cost, scheme, quantity):
    global input_EAR, EAR, EAR_11, final_optimized_cost, input_UTL, opStatus
    if scheme is None:
        input_cost = input_cost[input_cost["Food_Name"] != "Milk powder"]
    print('input-cost-check', input_cost)
    model = LpProblem("Optimal_diet_plan", LpMinimize)
    input_cost = input_cost.sort_values(['Food_Name'])
    variable_names = input_cost['Food_Name']
    DV_variables = LpVariable.matrix("X", variable_names, cat="Continuous", lowBound=0)

    allocation = np.array(DV_variables)
    input_cost = input_cost  # .sort_values(['Food'])
    Cost = input_cost["input_cost"]
    cost_matrix = Cost
    obj_func = lpSum(allocation * cost_matrix)
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

    # input EARRDA2020 DATASET----

    EARfull = pd.read_csv("Nutrition_gap_RDA_HCM.csv", encoding='unicode_escape')
    if Age_group == "child(4-6)yrs":
        input_EAR = EARfull["Gap4_6yrs"]
    if Age_group == "pregnant":
        input_EAR = EARfull["Gap_preg"]
    if Age_group == "lactation":
        input_EAR = EARfull["Gap_lact"]

    Lab = EARfull["Lab"]
    input_EAR = pd.concat([input_EAR, Lab], axis=1, ignore_index=True)
    input_EAR = input_EAR.rename(columns={0: "EAR", 1: "Lab"})
    EAR = input_EAR
    if scheme is None:
        milk = pd.read_csv("Milk_powder.csv", encoding='unicode_escape')
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

    ##min value for each nutrition
    EAR = EAR.loc[EAR["Lab"].isin(F.columns)]
    b = EAR["EAR"]
    b.index = EAR["Lab"]

    A = A.fillna(0)

    # ===================================#
    ######Selection----------------

    # nutrient<-c('Energy','Protein','Fat','Fibre','Iron','Calcium','Zinc','Iodine','Magnesium','VB12','VA','VB1','VB2','VB3','VB6','Folate','VC')
    nutrient = ['Energy', 'Protein', "Fat", 'Iron', 'Calcium', 'Zinc', 'Folate']

    mm = pd.DataFrame(A.loc[A.index.isin(nutrient)])
    AA = mm

    for i in range(len(AA)):
        model += lpSum(AA.iloc[i] * allocation) >= b.iloc[i]

    for i in range(len(allocation)):
        model += lpSum(allocation[i]) >= 0

    ######TUL ##########
    TULfull = pd.read_csv("UTL for nutritions_HCM.csv", encoding='unicode_escape')
    if (Age_group == "child(4-6)yrs"):
        input_UTL = TULfull["UTL4_6yrs"]
    if (Age_group == "pregnant"):
        input_UTL = TULfull["UTL_preg"]
    if (Age_group == "lactation"):
        input_UTL = TULfull["UTL_lact"]

    Lab = TULfull["Lab"]
    input_UTL = pd.concat([input_UTL, Lab], axis=1, ignore_index=True)
    input_UTL = input_UTL.rename(columns={0: "UTL", 1: "Lab"})
    UTL = input_UTL
    if (Age_group == "child(4-6)yrs"):
        UTL = UTL[UTL["Lab"] != "Folate"]

    UTL = UTL.loc[UTL["Lab"].isin(F.columns)]
    b1 = UTL["UTL"]
    b1.index = UTL["Lab"]

    A = A.fillna(0)

    # ===================================#
    ######Selection----------------

    # nutrient<-c('Energy','Protein','Fat','Fibre','Iron','Calcium','Zinc','Iodine','Magnesium','VB12','VA','VB1','VB2','VB3','VB6','Folate','VC')
    nutrient = UTL["Lab"]

    mm = pd.DataFrame(A.loc[A.index.isin(nutrient)])
    AA = mm

    for i in range(len(AA)):
        model += lpSum(AA.iloc[i] * allocation) <= b1.iloc[i]

    #####constraints for upper limit for fat based on energy

    model += lpSum(A.iloc[2,] * allocation) <= lpSum(A.iloc[0,] * allocation) * 0.40 / 9

    if Age_group == "child(4-6)yrs":
        model += lpSum(allocation * cost_matrix) <= 8

    if (Age_group == "pregnant") or (Age_group == "lactation"):
        model += lpSum(allocation * cost_matrix) <= 21

    # sugar*********************
    # Sugar as 10% of total energy
    # 1 g of carbohydrate (sugar) = 4 kcal

    if "Sugar" in tuple(Foodgroup):
        a = np.zeros((len(AA.columns)))
        a[AA.columns == "Sugar"] = 1
        a[AA.columns == "Jaggery"] = 1
        MK = EAR[EAR["Lab"] == "Energy"]
        usugar = (0.1 * MK["EAR"]) / 4
        a = ((4 * a) - (0.08 * A.iloc[0,]))
        model += lpSum(a * allocation) <= usugar

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
        a[AA.columns == Pulse[i]] = -2

    model += lpSum(a * allocation) == 0

    k = data[data["Food_Group"] == "Cereals"]
    fcereals = k["Food_Name"]
    fcereals.index = range(0, len(fcereals), 1)

    k = data[data["Food_Group"] == "Pulses"]
    fpulse = k["Food_Name"]
    fpulse.index = range(0, len(fpulse), 1)
    cereals = [1 / len(fcereals)] * len(fcereals)
    pulse = [1 / len(fpulse)] * len(fpulse)

    ##fix minimum limit
    a = np.zeros((len(AA.columns)))
    for i in range(len(fcereals)):
        a[AA.columns == fcereals[i]] = 1
    model += lpSum(a * allocation) >= 20

    a = np.zeros((len(AA.columns)))
    for i in range(len(fpulse)):
        a[AA.columns == fpulse[i]] = 1
    model += lpSum(a * allocation) >= 20

    # model += lpSum(allocation) <= 500/3 + (500/3) *0.4
    if Age_group == "child(4-6)yrs":
        model += lpSum(allocation) <= 250
    # model += lpSum(allocation) >= 500/3  - (500/3) *0.2

    if Age_group == "pregnant":
        model += lpSum(allocation) <= 1350 / 3 + 1350 / 3 * 0.3

    if Age_group == "lactation":
        model += lpSum(allocation) <= 1640 / 3 + 1640 / 3 * 0.3

    if "Egg" in tuple(Foodgroup):
        a = np.zeros((len(AA.columns)))
        a[AA.columns == "Egg"] = 1
        model += lpSum(a * allocation) <= 45

    if "Egg" in tuple(Foodgroup):
        a = np.zeros((len(AA.columns)))
        a[AA.columns == "Egg"] = 1
        model += lpSum(a * allocation) >= 15

    if "Fruits" in tuple(Foodgroup):
        a = np.zeros((len(AA.columns)))
        a[AA.columns == "Banana"] = 1
        model += lpSum(a * allocation) <= 20

    if "Oil" in tuple(Foodgroup):
        a = np.zeros((len(AA.columns)))
        a[AA.columns == "Oil"] = 1
        a[AA.columns == "Ghee"] = 1
        model += lpSum(a * allocation) >= 5

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
    opStatus = LpStatus[model.status]
    print('faaOpstatus:', opStatus)
    input_cost_v1 = input_cost.sort_values(["Food_Name"])
    costperitem = np.array(input_cost_v1["input_cost"])
    quan = np.array(final_out["Amount"])
    c_1 = costperitem * quan
    final_out["cost"] = c_1
    final_out["Cost (per Kg)"] = costperitem * 1000
    if scheme is None:
        final_out.loc[len(final_out.index)] = ['Milk powder', int(quantity), 'Milk powder', 0, 0]
    return final_out


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
    print(ou)
    return ou


def VEGNUTCAL(quantity_food):
    Food = quantity_food["Food_Name"]
    Food = Food.sort_values()

    data = pd.read_csv("Additional_Nutrition_sheet_ICDS_HCM.csv", encoding='unicode_escape')

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
    A2 = np.array(pd.to_numeric(quantity_food["Amount"]))

    nut_out = pd.DataFrame(np.dot(A1, A2))

    ou = pd.concat([nut_out, nutritions], axis=1, ignore_index=False)
    ou.columns = ["Amount", "Nutritions"]

    ou = np.round(ou, 1)
    print(ou)
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
    final_out_lessKcal["Cost (per Kg)"] = costperitem * 1000
    if scheme is None:
        final_out_lessKcal.loc[len(final_out_lessKcal.index)] = ['Milk powder', int(quantity), 'Milk powder', 0, 0]
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
    final_out["Cost (per Kg)"] = costperitem * 1000
    if scheme is None:
        final_out.loc[len(final_out.index)] = ['Milk powder', quantity, 'Milk powder', 0, 0]
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
            request.session.pop("preSchool", None)
            request.session.pop("pregnantFAA", None)
            request.session.pop("lactatingFAA", None)
            return render(request, 'icds/category.html')

        elif query == 'FAA':
            request.session.pop("infant", None)
            request.session.pop("toddler", None)
            request.session.pop("pregnant", None)
            request.session.pop("lactating", None)
            request.session.pop("preSchool", None)
            request.session.pop("pregnantFAA", None)
            request.session.pop("lactatingFAA", None)
            return render(request, 'icds/categoryFAA.html')

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
            request.session.pop("preSchool", None)
            request.session.pop("pregnantFAA", None)
            request.session.pop("lactatingFAA", None)

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

        elif query == 'FAA':
            context = {
                'data': request.POST,
                'has_error': False
            }
            request.session.pop("infant", None)
            request.session.pop("toddler", None)
            request.session.pop("pregnant", None)
            request.session.pop("lactating", None)
            request.session.pop("preSchool", None)
            request.session.pop("pregnantFAA", None)
            request.session.pop("lactatingFAA", None)

            preSchool = request.POST.get('preSchool', None)
            pregnantFAA = request.POST.get('pregnantFAA', None)
            lactatingFAA = request.POST.get('lactatingFAA', None)

            if context['has_error']:
                return render(request, 'icds/index.html', context=context,
                              status=status.HTTP_400_BAD_REQUEST)

            request.session['preSchool'] = int(preSchool)
            request.session['pregnantFAA'] = int(pregnantFAA)
            request.session['lactatingFAA'] = int(lactatingFAA)

            return redirect('Anganwadi_FoodSelection')

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

        elif request.session['query'] == 'FAA':
            if 'preSchool' not in request.session:
                return redirect('home')
            preSchool = request.session['preSchool']
            pregnantFAA = request.session['pregnantFAA']
            lactatingFAA = request.session['lactatingFAA']

            if preSchool <= 0 and pregnantFAA <= 0 and lactatingFAA <= 0:
                messages.add_message(request, messages.ERROR, 'Please enter at least one! ')
                return render(request, 'icds/categoryFAA.html')
            print(preSchool, pregnantFAA, lactatingFAA)
            return render(request, 'icds/foodSelectionFAA.html',
                          {'preSchool': preSchool, 'pregnantFAA': pregnantFAA, 'lactatingFAA': lactatingFAA})

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

        elif request.session['query'] == 'FAA':
            if 'preSchool' not in request.session:
                messages.add_message(request, messages.ERROR, 'Please enter the number of people again! ')
                return redirect('category')

            Cereals = request.POST.getlist('Cereals', None)
            Pulses = request.POST.getlist('Pulses', None)
            Others = request.POST.getlist('Others', None)
            milkpowder = request.POST.getlist('milkpowder', None)
            print('MILK-POWDER', milkpowder)
            scheme = request.POST.get('scheme', None)
            milkPowderQuantity = request.POST.get('milkPowderQuantity', None)

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
            request.session['pregnantCereals'] = pregnantCereals
            request.session['pregnantPulses'] = pregnantPulses
            request.session['pregnantOthers'] = pregnantOthers
            request.session['lactatingCereals'] = lactatingCereals
            request.session['lactatingPulses'] = lactatingPulses
            request.session['lactatingOthers'] = lactatingOthers
            request.session['milkpowder'] = milkpowder
            request.session['scheme'] = scheme
            request.session['milkPowderQuantity'] = milkPowderQuantity

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

        # FAA group
        elif request.session['query'] == 'FAA':
            if 'preSchool' not in request.session:
                return redirect('home')

            preSchool = request.session['preSchool']
            pregnantFAA = request.session['pregnantFAA']
            lactatingFAA = request.session['lactatingFAA']

            Cereals = request.session['Cereals']
            Pulses = request.session['Pulses']
            Others = request.session['Others']
            milkpowder = request.session['milkpowder']
            preSchoolFood = itertools.chain(Cereals, Pulses, Others, milkpowder)
            preSchoolFood = list(preSchoolFood)
            request.session['preSchoolFood'] = preSchoolFood

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

            resultCereals = sorted(np.unique(Cereals + pregnantCereals + lactatingCereals))
            print(resultCereals)
            resultPulses = sorted(np.unique(Pulses + pregnantPulses + lactatingPulses))
            print(resultPulses)
            resultOthers = sorted(np.unique(Others + pregnantOthers + lactatingOthers))
            print(resultOthers)
            resultMilkPowder = sorted(
                np.unique(milkpowder + pregnantmilkpowder + lactatingmilkpowder))
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

            return render(request, 'icds/foodCostFAA.html',
                          {'preSchool': preSchool, 'pregnantFAA': pregnantFAA, 'lactatingFAA': lactatingFAA,
                           'preSchoolFood': preSchoolFood, 'pregnantFood': pregnantFood,
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
        if request.session['query'] == 'THR':
            if 'infant' not in request.session:
                return redirect('home')

            infant = request.session['infant']
            toddler = request.session['toddler']
            pregnant = request.session['pregnant']
            lactating = request.session['lactating']
            Cereals = request.session.get('Cereals', None)
            Pulses = request.session.get('Pulses', None)
            Others = request.session.get('Others', None)
            scheme = request.session.get('scheme', None)

            milkPowderQuantity = request.session.get('milkPowderQuantity', None)
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

            return render(request, 'icds/result.html',
                          {'module': 'THR', 'infant': infant, 'toddler': toddler, 'pregnant': pregnant,
                           'lactating': lactating,
                           'Cereals': Cereals, 'Pulses': Pulses, 'Others': Others, 'cereal_prop': cereal_prop,
                           'pulse_prop': pulse_prop, 'other_prop': other_prop, 'final_q': final_q,
                           'final_optimized_cost': final_optimized_cost, 'opStatus': opStatus
                           })

        if request.session['query'] == 'FAA':
            if 'preSchool' not in request.session:
                return redirect('home')

            preSchool = request.session['preSchool']
            pregnantFAA = request.session['pregnantFAA']
            lactatingFAA = request.session['lactatingFAA']
            Cereals = request.session['Cereals']
            Pulses = request.session['Pulses']
            Others = request.session['Others']
            scheme = request.session['scheme']
            milkPowderQuantity = request.session['milkPowderQuantity']
            if milkPowderQuantity is not None:
                milkPowderQuantity = int(milkPowderQuantity)
            request.session.pop("pregnant_Qnt_list", None)
            cereal_prop = []
            pulse_prop = []
            other_prop = []
            final_q = ''
            if request.session['preSchool'] > 0:
                cost_list = request.session['cost_list']
                for k, v in cost_list.items():
                    cost_list[k] = float(v)

                cost_list = pd.DataFrame(cost_list.items(), columns=['Food_Name', 'input_cost'])
                print(cost_list)

                preSchoolFood = request.session['preSchoolFood']
                print(preSchoolFood)
                preSchoolFoodCost = cost_list[cost_list['Food_Name'].isin(preSchoolFood)]
                print(preSchoolFoodCost)

                preSchoolFoodCost['input_cost'] = preSchoolFoodCost['input_cost'] / 1000
                Age_group = "4yrs-6yrs"
                Food = preSchoolFoodCost['Food_Name']

            return render(request, 'icds/resultFAA.html',
                          {'module': 'FAA', 'preSchool': preSchool, 'pregnantFAA': pregnantFAA,
                           'lactatingFAA': lactatingFAA,
                           'Cereals': Cereals, 'Pulses': Pulses, 'Others': Others, 'cereal_prop': cereal_prop,
                           'pulse_prop': pulse_prop, 'other_prop': other_prop
                           })

        return render(request, 'icds/index.html')


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

            if opStatus == 'Infeasible' and opStatus_lessKcal == 'Infeasible':
                i_Infeasible = render_to_string('icds/infeasible.html')
                return JsonResponse({'data': i_Infeasible})
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
            if opStatus == 'Infeasible':
                t_Infeasible = render_to_string('icds/infeasible.html')
                return JsonResponse({'data': t_Infeasible})
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

        request.session.pop("lactating_Qnt_list", None)
        request.session.pop("pregnant_Qnt_list", None)
        request.session.pop("preSchool_Qnt_list", None)

        cereal_prop_pregnant = []
        pulse_prop_pregnant = []
        other_prop_pregnant = []
        milk_prop_pregnant = []

        if request.session.get('pregnant', None) or request.session.get('pregnantFAA', None):
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

            query = request.session['query']
            print(query)
            if query == 'THR':
                final_out_pregnant = LPPWOVAR(Age_group, Food, pregnantFoodCost, pregnantscheme,
                                              pregnantmilkPowderQuantity)
            elif query == 'FAA':
                final_out_pregnant = LPPWOVARHCM(Age_group, Food, pregnantFoodCost, pregnantscheme,
                                                 pregnantmilkPowderQuantity)

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
            print(opStatus)
            if opStatus == 'Infeasible':
                p_Infeasible = render_to_string('icds/infeasible.html')
                return JsonResponse({'data': p_Infeasible})
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
        request.session.pop("lactating_Qnt_list", None)
        request.session.pop("pregnant_Qnt_list", None)
        request.session.pop("preSchool_Qnt_list", None)
        cereal_prop_lactating = []
        pulse_prop_lactating = []
        other_prop_lactating = []
        milk_prop_lactating = []

        if request.session.get('lactating', None) or request.session.get('lactatingFAA', None):

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

            query = request.session['query']
            print(query)
            if query == 'THR':
                final_out_lactating = LPPWOVAR(Age_group, Food, lactatingFoodCost, lactatingscheme,
                                               lactatingmilkPowderQuantity)
            elif query == 'FAA':
                final_out_lactating = LPPWOVARHCM(Age_group, Food, lactatingFoodCost, lactatingscheme,
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
            if opStatus == 'Infeasible':
                lw_Infeasible = render_to_string('icds/infeasible.html')
                return JsonResponse({'data': lw_Infeasible})

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

    if category_id == '4 yrs to 6 yrs':
        Cereals = request.session['Cereals']
        Pulses = request.session['Pulses']
        Others = request.session['Others']
        milkpowder = request.session['milkpowder']
        scheme = request.session['scheme']
        milkPowderQuantity = request.session['milkPowderQuantity']
        if milkPowderQuantity is not None:
            milkPowderQuantity = int(milkPowderQuantity)
        request.session.pop("lactating_Qnt_list", None)
        request.session.pop("pregnant_Qnt_list", None)
        request.session.pop("preSchool_Qnt_list", None)
        cereal_prop = []
        pulse_prop = []
        other_prop = []
        milk_prop = []

        if request.session['preSchool'] > 0:
            cost_list = request.session['cost_list']
            for k, v in cost_list.items():
                cost_list[k] = float(v)

            cost_list = pd.DataFrame(cost_list.items(), columns=['Food_Name', 'input_cost'])
            print(cost_list)

            preSchoolFood = request.session['preSchoolFood']
            print(preSchoolFood)
            infantFoodCost = cost_list[cost_list['Food_Name'].isin(preSchoolFood)]
            print(infantFoodCost)

            infantFoodCost['input_cost'] = infantFoodCost['input_cost'] / 1000
            Age_group = "child(4-6)yrs"
            Food = infantFoodCost['Food_Name']

            final_out_infant = LPPWOVARHCM(Age_group, Food, infantFoodCost, scheme, milkPowderQuantity)

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

            if opStatus == 'Infeasible':
                i_Infeasible = render_to_string('icds/infeasible.html')
                return JsonResponse({'data': i_Infeasible})
            i = render_to_string('icds/resultPreSchool.html',
                                 {
                                     'Cereals': Cereals, 'Pulses': Pulses, 'Others': Others, 'milkpowder': milkpowder,
                                     'cereal_prop': cereal_prop, 'pulse_prop': pulse_prop, 'other_prop': other_prop,
                                     'milk_prop': milk_prop,
                                     'final_q': final_q, 'final_optimized_cost': final_optimized_cost,
                                     'opStatus': opStatus, 'scheme': scheme
                                 })
            return JsonResponse({'data': i})


# getting pdf for result
class GetPdf(View):
    def get(self, request):
        if ('query' not in request.session) and ('infant' not in request.session):
            return redirect('home')

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

        inft_Status = ''
        inft_Status_lessKcal = ''
        todd_Status = ''
        preg_Status = ''
        lact_Status = ''

        cost_list = request.session.get('cost_list', None)
        print(cost_list)
        if 'cost_list' not in request.session:
            return redirect('home')

        for k, v in cost_list.items():
            cost_list[k] = float(v)

        cost_list = pd.DataFrame(cost_list.items(), columns=['Food_Name', 'input_cost'])
        print(cost_list)

        if request.session.get('query', None) == 'THR':
            infant = request.session.get('infant', None)
            toddler = request.session.get('toddler', None)
            pregnant = request.session.get('pregnant', None)
            lactating = request.session.get('lactating', None)

            if infant > 0:
                scheme = request.session.get('scheme', None)
                milkPowderQuantity = request.session.get('milkPowderQuantity', None)
                if milkPowderQuantity is not None:
                    milkPowderQuantity = int(milkPowderQuantity)
                infantFood = request.session.get('infantFood', None)
                print(infantFood)
                infantFoodCost = cost_list[cost_list['Food_Name'].isin(infantFood)]
                print(infantFoodCost)

                infantFoodCost['input_cost'] = infantFoodCost['input_cost'] / 1000
                Age_group = "6-12 months"
                Food = infantFoodCost['Food_Name']

                final_out_infant = LPPWOVAR(Age_group, Food, infantFoodCost, scheme, milkPowderQuantity)
                inft_Status = opStatus
                inft_nutrition = NUTCAL(final_out_infant)

                final_out_infant = final_out_infant[final_out_infant['Amount'] > 0]
                final_out_infant.columns = ['Food Name', 'Per Person Intake (gm)', 'Food Group', 'Per Person Cost (Rs)',
                                            'Cost (per kg)']
                final_out_infant.reset_index(inplace=True, drop=True)
                final_out_infant = final_out_infant.drop(['Food Group'], axis=1)
                final_out_infant['Total Quantity (gm)'] = final_out_infant['Per Person Intake (gm)'] * infant
                final_out_infant['Total Cost (Rs)'] = final_out_infant['Per Person Cost (Rs)'] * infant
                final_out_infant = final_out_infant[
                    ["Food Name", "Cost (per kg)", "Per Person Intake (gm)", "Per Person Cost (Rs)",
                     "Total Quantity (gm)",
                     "Total Cost (Rs)"]]
                inft_data = final_out_infant.to_html(classes='mystyle', index=False)
                inft_total = final_out_infant['Total Cost (Rs)'].sum()
                inft_total = inft_total.round(2)

                inft_perc, inft_fat_perc, inft_other_nut = Percentagecalculation(inft_nutrition, Age_group)
                inft_perc = inft_perc.to_html(classes='mystyle', index=False)
                inft_other_nut = inft_other_nut.to_html(classes='mystyle', index=False)

                # 6 month to 1 year 250Kcal
                final_out_infant_lessKcal = LPPWOVAR_LESSKCAL(Age_group, Food, infantFoodCost, scheme,
                                                              milkPowderQuantity)
                inft_Status_lessKcal = opStatus
                inft_nutrition_lessKcal = NUTCAL(final_out_infant_lessKcal)

                final_out_infant_lessKcal = final_out_infant_lessKcal[final_out_infant_lessKcal['Amount'] > 0]
                final_out_infant_lessKcal.columns = ['Food Name', 'Per Person Intake (gm)', 'Food Group',
                                                     'Per Person Cost (Rs)', 'Cost (per kg)']
                final_out_infant_lessKcal.reset_index(inplace=True, drop=True)
                final_out_infant_lessKcal = final_out_infant_lessKcal.drop(['Food Group'], axis=1)
                final_out_infant_lessKcal['Total Quantity (gm)'] = final_out_infant_lessKcal[
                                                                       'Per Person Intake (gm)'] * infant
                final_out_infant_lessKcal['Total Cost (Rs)'] = final_out_infant_lessKcal[
                                                                   'Per Person Cost (Rs)'] * infant
                final_out_infant_lessKcal = final_out_infant_lessKcal[
                    ["Food Name", "Cost (per kg)", "Per Person Intake (gm)", "Per Person Cost (Rs)",
                     "Total Quantity (gm)",
                     "Total Cost (Rs)"]]
                inft_data_lessKcal = final_out_infant_lessKcal.to_html(classes='mystyle', index=False)

                inft_total_lessKcal = final_out_infant_lessKcal['Total Cost (Rs)'].sum()
                inft_total_lessKcal = inft_total_lessKcal.round(2)

                inft_perc_lessKcal, inft_fat_perc_lessKcal, inft_other_nut_lessKcal = Percentagecalculation(
                    inft_nutrition_lessKcal, Age_group)
                inft_perc_lessKcal = inft_perc_lessKcal.to_html(classes='mystyle', index=False)
                inft_other_nut_lessKcal = inft_other_nut_lessKcal.to_html(classes='mystyle', index=False)

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

                final_out_toddler = LPPWOVAR(Age_group, Food, toddlersFoodCost, toddlersscheme,
                                             toddlersmilkPowderQuantity)
                todd_Status = opStatus
                todd_nutrition = NUTCAL(final_out_toddler)

                final_out_toddler = final_out_toddler[final_out_toddler['Amount'] > 0]
                final_out_toddler.columns = ['Food Name', 'Per Person Intake (gm)', 'Food Group',
                                             'Per Person Cost (Rs)',
                                             'Cost (per kg)']
                final_out_toddler.reset_index(inplace=True, drop=True)
                final_out_toddler = final_out_toddler.drop(['Food Group'], axis=1)
                final_out_toddler['Total Quantity (gm)'] = final_out_toddler['Per Person Intake (gm)'] * toddler
                final_out_toddler['Total Cost (Rs)'] = final_out_toddler['Per Person Cost (Rs)'] * toddler
                final_out_toddler = final_out_toddler[
                    ["Food Name", "Cost (per kg)", "Per Person Intake (gm)", "Per Person Cost (Rs)",
                     "Total Quantity (gm)",
                     "Total Cost (Rs)"]]
                todd_data = final_out_toddler.to_html(classes='mystyle', index=False)
                todd_total = final_out_toddler['Total Cost (Rs)'].sum()
                todd_total = todd_total.round(2)

                todd_perc, todd_fat_perc, todd_other_nut = Percentagecalculation(todd_nutrition, Age_group)
                todd_perc = todd_perc.to_html(classes='mystyle', index=False)
                todd_other_nut = todd_other_nut.to_html(classes='mystyle', index=False)

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

                final_out_pregnant = LPPWOVAR(Age_group, Food, pregnantFoodCost, pregnantscheme,
                                              pregnantmilkPowderQuantity)
                preg_Status = opStatus
                preg_nutrition = NUTCAL(final_out_pregnant)

                final_out_pregnant = final_out_pregnant[final_out_pregnant['Amount'] > 0]
                final_out_pregnant.columns = ['Food Name', 'Per Person Intake (gm)', 'Food Group',
                                              'Per Person Cost (Rs)',
                                              'Cost (per kg)']
                final_out_pregnant.reset_index(inplace=True, drop=True)
                final_out_pregnant = final_out_pregnant.drop(['Food Group'], axis=1)
                final_out_pregnant['Total Quantity (gm)'] = final_out_pregnant['Per Person Intake (gm)'] * pregnant
                final_out_pregnant['Total Cost (Rs)'] = final_out_pregnant['Per Person Cost (Rs)'] * pregnant
                final_out_pregnant = final_out_pregnant[
                    ["Food Name", "Cost (per kg)", "Per Person Intake (gm)", "Per Person Cost (Rs)",
                     "Total Quantity (gm)",
                     "Total Cost (Rs)"]]
                preg_data = final_out_pregnant.to_html(classes='mystyle', index=False)

                preg_total = final_out_pregnant['Total Cost (Rs)'].sum()
                preg_total = preg_total.round(2)

                preg_perc, preg_fat_perc, preg_other_nut = Percentagecalculation(preg_nutrition, Age_group)
                preg_perc = preg_perc.to_html(classes='mystyle', index=False)
                preg_other_nut = preg_other_nut.to_html(classes='mystyle', index=False)

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
                lact_Status = opStatus
                lact_nutrition = NUTCAL(final_out_lactating)

                final_out_lactating = final_out_lactating[final_out_lactating['Amount'] > 0]
                final_out_lactating.columns = ['Food Name', 'Per Person Intake (gm)', 'Food Group',
                                               'Per Person Cost (Rs)',
                                               'Cost (per kg)']
                final_out_lactating.reset_index(inplace=True, drop=True)
                final_out_lactating = final_out_lactating.drop(['Food Group'], axis=1)
                final_out_lactating['Total Quantity (gm)'] = final_out_lactating['Per Person Intake (gm)'] * lactating
                final_out_lactating['Total Cost (Rs)'] = final_out_lactating['Per Person Cost (Rs)'] * lactating
                final_out_lactating = final_out_lactating[
                    ["Food Name", "Cost (per kg)", "Per Person Intake (gm)", "Per Person Cost (Rs)",
                     "Total Quantity (gm)",
                     "Total Cost (Rs)"]]
                lact_data = final_out_lactating.to_html(classes='mystyle', index=False, )

                lact_total = final_out_lactating['Total Cost (Rs)'].sum()
                lact_total = lact_total.round(2)
                lact_perc, lact_fat_perc, lact_other_nut = Percentagecalculation(lact_nutrition, Age_group)
                lact_perc = lact_perc.to_html(classes='mystyle', index=False)
                lact_other_nut = lact_other_nut.to_html(classes='mystyle', index=False)

            params = {
                'module': 'THR',
                'today': datetime.now(),
                'infant': infant, 'toddler': toddler, 'pregnant': pregnant, 'lactating': lactating,

                'inft_data': inft_data, 'inft_perc': inft_perc, 'inft_fat_perc': inft_fat_perc,
                'inft_other_nut': inft_other_nut, 'inft_Status': inft_Status, 'inft_total': inft_total,

                'lact_data': lact_data, 'lact_total': lact_total, 'lact_perc': lact_perc,
                'lact_other_nut': lact_other_nut, 'lact_fat_perc': lact_fat_perc, 'lact_Status': lact_Status,

                'preg_total': preg_total, 'preg_perc': preg_perc, 'preg_fat_perc': preg_fat_perc,
                'preg_other_nut': preg_other_nut, 'preg_Status': preg_Status, 'preg_data': preg_data,

                'todd_data': todd_data, 'todd_perc': todd_perc, 'todd_fat_perc': todd_fat_perc,
                'todd_other_nut': todd_other_nut, 'todd_total': todd_total, 'todd_Status': todd_Status,

                'inft_data_lessKcal': inft_data_lessKcal, 'inft_perc_lessKcal': inft_perc_lessKcal,
                'inft_fat_perc_lessKcal': inft_fat_perc_lessKcal, 'inft_other_nut_lessKcal': inft_other_nut_lessKcal,
                'inft_Status_lessKcal': inft_Status_lessKcal, 'inft_total_lessKcal': inft_total_lessKcal,

            }
            return Render.render('icds/pdf.html', params)

        elif request.session.get('query', None) == 'FAA':
            preSchool = request.session['preSchool']
            pregnantFAA = request.session['pregnantFAA']
            lactatingFAA = request.session['lactatingFAA']

            if preSchool > 0:
                scheme = request.session['scheme']
                milkPowderQuantity = request.session['milkPowderQuantity']
                if milkPowderQuantity is not None:
                    milkPowderQuantity = int(milkPowderQuantity)
                preSchoolFood = request.session['preSchoolFood']
                print(preSchoolFood)
                preSchoolFoodCost = cost_list[cost_list['Food_Name'].isin(preSchoolFood)]
                print(preSchoolFoodCost)

                preSchoolFoodCost['input_cost'] = preSchoolFoodCost['input_cost'] / 1000
                Age_group = "child(4-6)yrs"
                Food = preSchoolFoodCost['Food_Name']

                final_out_preSchool = LPPWOVARHCM(Age_group, Food, preSchoolFoodCost, scheme, milkPowderQuantity)
                inft_Status = opStatus
                inft_nutrition = NUTCAL(final_out_preSchool)

                final_out_preSchool = final_out_preSchool[final_out_preSchool['Amount'] > 0]
                final_out_preSchool.columns = ['Food Name', 'Per Person Intake (gm)', 'Food Group',
                                               'Per Person Cost (Rs)',
                                               'Cost (per kg)']
                final_out_preSchool.reset_index(inplace=True, drop=True)
                final_out_preSchool = final_out_preSchool.drop(['Food Group'], axis=1)
                final_out_preSchool['Total Quantity (gm)'] = final_out_preSchool['Per Person Intake (gm)'] * preSchool
                final_out_preSchool['Total Cost (Rs)'] = final_out_preSchool['Per Person Cost (Rs)'] * preSchool
                final_out_preSchool = final_out_preSchool[
                    ["Food Name", "Cost (per kg)", "Per Person Intake (gm)", "Per Person Cost (Rs)",
                     "Total Quantity (gm)",
                     "Total Cost (Rs)"]]
                inft_total = final_out_preSchool['Total Cost (Rs)'].sum()
                inft_total = inft_total.round(2)
                inft_perc, inft_fat_perc, inft_other_nut = PercentagecalculationHCM(inft_nutrition, Age_group)

                preSchool_Qnt_list = request.session.get('preSchool_Qnt_list', None)
                if preSchool_Qnt_list is not None and len(preSchool_Qnt_list) > 0:
                    preSchool_Qnt_df = pd.DataFrame(preSchool_Qnt_list.items())
                    preSchool_Qnt_df.columns = ['Food_Name', 'Amount']
                    nutrition_calc = VEGNUTCAL(preSchool_Qnt_df)
                    print(nutrition_calc)

                    preSchool_Qnt_df.columns = ['Food Name', 'Per Person Intake (gm)']
                    preSchool_Qnt_df['Total Quantity (gm)'] = preSchool_Qnt_df['Per Person Intake (gm)'] * preSchool

                    final_out_preSchool = final_out_preSchool.drop(
                        ['Cost (per kg)', 'Per Person Cost (Rs)', 'Total Cost (Rs)'], axis=1)

                    final_out_preSchool = pd.concat([final_out_preSchool, preSchool_Qnt_df], axis=0)
                    print(final_out_preSchool)

                    df_add = pd.DataFrame()
                    df_add["Nutritions"] = inft_nutrition["Nutritions"]
                    df_add["Amount"] = inft_nutrition["Amount"] + nutrition_calc["Amount"]
                    print(df_add)
                    inft_perc, inft_fat_perc, inft_other_nut = PercentagecalculationHCM(df_add, Age_group)

                inft_data = final_out_preSchool.to_html(classes='mystyle', index=False)

                inft_perc = inft_perc.to_html(classes='mystyle', index=False)
                inft_other_nut = inft_other_nut.to_html(classes='mystyle', index=False)

                # check if none is selected

            if pregnantFAA > 0:
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

                final_out_pregnant = LPPWOVARHCM(Age_group, Food, pregnantFoodCost, pregnantscheme,
                                                 pregnantmilkPowderQuantity)
                preg_Status = opStatus
                preg_nutrition = NUTCAL(final_out_pregnant)

                final_out_pregnant = final_out_pregnant[final_out_pregnant['Amount'] > 0]
                final_out_pregnant.columns = ['Food Name', 'Per Person Intake (gm)', 'Food Group',
                                              'Per Person Cost (Rs)',
                                              'Cost (per kg)']
                final_out_pregnant.reset_index(inplace=True, drop=True)
                final_out_pregnant = final_out_pregnant.drop(['Food Group'], axis=1)
                final_out_pregnant['Total Quantity (gm)'] = final_out_pregnant['Per Person Intake (gm)'] * pregnantFAA
                final_out_pregnant['Total Cost (Rs)'] = final_out_pregnant['Per Person Cost (Rs)'] * pregnantFAA
                final_out_pregnant = final_out_pregnant[
                    ["Food Name", "Cost (per kg)", "Per Person Intake (gm)", "Per Person Cost (Rs)",
                     "Total Quantity (gm)",
                     "Total Cost (Rs)"]]

                preg_total = final_out_pregnant['Total Cost (Rs)'].sum()
                preg_total = preg_total.round(2)

                preg_perc, preg_fat_perc, preg_other_nut = PercentagecalculationHCM(preg_nutrition, Age_group)

                pregnant_Qnt_list = request.session.get('pregnant_Qnt_list', None)
                if pregnant_Qnt_list is not None and len(pregnant_Qnt_list) > 0:
                    pregnant_Qnt_df = pd.DataFrame(pregnant_Qnt_list.items())
                    pregnant_Qnt_df.columns = ['Food_Name', 'Amount']
                    nutrition_calc = VEGNUTCAL(pregnant_Qnt_df)
                    print(nutrition_calc)

                    pregnant_Qnt_df.columns = ['Food Name', 'Per Person Intake (gm)']
                    pregnant_Qnt_df['Total Quantity (gm)'] = pregnant_Qnt_df['Per Person Intake (gm)'] * pregnantFAA

                    final_out_pregnant = final_out_pregnant.drop(
                        ['Cost (per kg)', 'Per Person Cost (Rs)', 'Total Cost (Rs)'], axis=1)

                    final_out_pregnant = pd.concat([final_out_pregnant, pregnant_Qnt_df], axis=0)
                    print(final_out_pregnant)

                    df_add = pd.DataFrame()
                    df_add["Nutritions"] = preg_nutrition["Nutritions"]
                    df_add["Amount"] = preg_nutrition["Amount"] + nutrition_calc["Amount"]
                    print(df_add)
                    preg_perc, preg_fat_perc, preg_other_nut = PercentagecalculationHCM(df_add, Age_group)

                preg_data = final_out_pregnant.to_html(classes='mystyle', index=False)

                preg_perc = preg_perc.to_html(classes='mystyle', index=False)
                preg_other_nut = preg_other_nut.to_html(classes='mystyle', index=False)

            if lactatingFAA > 0:
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
                lact_Status = opStatus
                lact_nutrition = NUTCAL(final_out_lactating)

                final_out_lactating = final_out_lactating[final_out_lactating['Amount'] > 0]
                final_out_lactating.columns = ['Food Name', 'Per Person Intake (gm)', 'Food Group',
                                               'Per Person Cost (Rs)',
                                               'Cost (per kg)']
                final_out_lactating.reset_index(inplace=True, drop=True)
                final_out_lactating = final_out_lactating.drop(['Food Group'], axis=1)
                final_out_lactating['Total Quantity (gm)'] = final_out_lactating[
                                                                 'Per Person Intake (gm)'] * lactatingFAA
                final_out_lactating['Total Cost (Rs)'] = final_out_lactating['Per Person Cost (Rs)'] * lactatingFAA
                final_out_lactating = final_out_lactating[
                    ["Food Name", "Cost (per kg)", "Per Person Intake (gm)", "Per Person Cost (Rs)",
                     "Total Quantity (gm)",
                     "Total Cost (Rs)"]]

                lact_total = final_out_lactating['Total Cost (Rs)'].sum()
                lact_total = lact_total.round(2)
                lact_perc, lact_fat_perc, lact_other_nut = PercentagecalculationHCM(lact_nutrition, Age_group)

                lactating_Qnt_list = request.session.get('lactating_Qnt_list', None)
                if lactating_Qnt_list is not None and len(lactating_Qnt_list) > 0:
                    lactating_Qnt_df = pd.DataFrame(lactating_Qnt_list.items())
                    lactating_Qnt_df.columns = ['Food_Name', 'Amount']
                    nutrition_calc = VEGNUTCAL(lactating_Qnt_df)
                    print(nutrition_calc)

                    lactating_Qnt_df.columns = ['Food Name', 'Per Person Intake (gm)']
                    lactating_Qnt_df['Total Quantity (gm)'] = lactating_Qnt_df['Per Person Intake (gm)'] * lactatingFAA

                    final_out_lactating = final_out_lactating.drop(
                        ['Cost (per kg)', 'Per Person Cost (Rs)', 'Total Cost (Rs)'], axis=1)

                    final_out_lactating = pd.concat([final_out_lactating, lactating_Qnt_df], axis=0)
                    print(final_out_lactating)

                    df_add = pd.DataFrame()
                    df_add["Nutritions"] = lact_nutrition["Nutritions"]
                    df_add["Amount"] = lact_nutrition["Amount"] + nutrition_calc["Amount"]
                    print(df_add)
                    lact_perc, lact_fat_perc, lact_other_nut = PercentagecalculationHCM(df_add, Age_group)

                lact_data = final_out_lactating.to_html(classes='mystyle', index=False)

                lact_perc = lact_perc.to_html(classes='mystyle', index=False)
                lact_other_nut = lact_other_nut.to_html(classes='mystyle', index=False)

            params = {
                'module': 'FAA',
                'today': datetime.now(),
                'preSchool': preSchool, 'pregnant': pregnantFAA, 'lactating': lactatingFAA, 'preg_data': preg_data,
                'preg_total': preg_total, 'preg_perc': preg_perc, 'preg_fat_perc': preg_fat_perc,
                'preg_other_nut': preg_other_nut, 'preg_Status': preg_Status,
                'lact_total': lact_total, 'lact_data': lact_data, 'lact_perc': lact_perc,
                'lact_other_nut': lact_other_nut, 'lact_fat_perc': lact_fat_perc, 'lact_Status': lact_Status,
                'inft_data': inft_data, 'inft_perc': inft_perc, 'inft_fat_perc': inft_fat_perc,
                'inft_other_nut': inft_other_nut, 'inft_Status': inft_Status, 'inft_total': inft_total,
            }
            return Render.render('icds/pdf.html', params)


class Recipe(View):

    def get(self, request):
        if 'query' not in request.session:
            return redirect('home')
        if request.session['query'] == 'THR':
            return redirect('home')

        elif request.session['query'] == 'FAA':
            if 'preSchool' not in request.session:
                return redirect('home')
            preSchool = request.session.get('preSchool', None)
            pregnantFAA = request.session.get('pregnantFAA', None)
            lactatingFAA = request.session.get('lactatingFAA', None)
            print(preSchool, pregnantFAA, lactatingFAA)

            return render(request, 'icds/vegetableSelectionFAA.html',
                          {'preSchool': preSchool, 'pregnantFAA': pregnantFAA, 'lactatingFAA': lactatingFAA})

        return render(request, 'icds/index.html')

    def post(self, request):
        if 'query' not in request.session:
            return redirect('home')

        if request.session['query'] == 'THR':
            return redirect('home')

        elif request.session['query'] == 'FAA':

            preSchoolGlv = request.POST.getlist('preSchoolGlv', None)
            preSchoolRt = request.POST.getlist('preSchoolRt', None)
            preSchoolOveg = request.POST.getlist('preSchoolOveg', None)
            preSchoolFruit = request.POST.getlist('preSchoolFruit', None)
            preSchoolCs = request.POST.getlist('preSchoolCs', None)

            pregnantGlv = request.POST.getlist('pregnantGlv', None)
            pregnantRt = request.POST.getlist('pregnantRt', None)
            pregnantOveg = request.POST.getlist('pregnantOveg', None)
            pregnantFruit = request.POST.getlist('pregnantFruit', None)
            pregnantCs = request.POST.getlist('pregnantCs', None)

            lactatingGlv = request.POST.getlist('lactatingGlv', None)
            lactatingRt = request.POST.getlist('lactatingRt', None)
            lactatingOveg = request.POST.getlist('lactatingOveg', None)
            lactatingFruit = request.POST.getlist('lactatingFruit', None)
            lactatingCs = request.POST.getlist('lactatingCs', None)

            request.session['preSchoolGlv'] = preSchoolGlv
            request.session['preSchoolRt'] = preSchoolRt
            request.session['preSchoolOveg'] = preSchoolOveg
            request.session['preSchoolFruit'] = preSchoolFruit
            request.session['preSchoolCs'] = preSchoolCs

            request.session['pregnantGlv'] = pregnantGlv
            request.session['pregnantRt'] = pregnantRt
            request.session['pregnantOveg'] = pregnantOveg
            request.session['pregnantFruit'] = pregnantFruit
            request.session['pregnantCs'] = pregnantCs

            request.session['lactatingGlv'] = lactatingGlv
            request.session['lactatingRt'] = lactatingRt
            request.session['lactatingOveg'] = lactatingOveg
            request.session['lactatingFruit'] = lactatingFruit
            request.session['lactatingCs'] = lactatingCs
            return redirect('vegQuantity')

        return render(request, 'icds/index.html',
                      )


class VegQuantity(View):

    def get(self, request):
        if 'query' not in request.session:
            return redirect('home')
        if request.session['query'] == 'THR':
            return redirect('home')

        elif request.session['query'] == 'FAA':
            if 'preSchool' not in request.session:
                return redirect('home')

            preSchool = request.session.get('preSchool', None)
            pregnantFAA = request.session.get('pregnantFAA', None)
            lactatingFAA = request.session.get('lactatingFAA', None)

            preSchoolGlv = request.session.get('preSchoolGlv', None)
            preSchoolRt = request.session.get('preSchoolRt', None)
            preSchoolOveg = request.session.get('preSchoolOveg', None)
            preSchoolFruit = request.session.get('preSchoolFruit', None)
            preSchoolCs = request.session.get('preSchoolCs', None)

            pregnantGlv = request.session.get('pregnantGlv', None)
            pregnantRt = request.session.get('pregnantRt', None)
            pregnantOveg = request.session.get('pregnantOveg', None)
            pregnantFruit = request.session.get('pregnantFruit', None)
            pregnantCs = request.session.get('pregnantCs', None)

            lactatingGlv = request.session.get('lactatingGlv', None)
            lactatingRt = request.session.get('lactatingRt', None)
            lactatingOveg = request.session.get('lactatingOveg', None)
            lactatingFruit = request.session.get('lactatingFruit', None)
            lactatingCs = request.session.get('lactatingCs', None)

            return render(request, 'icds/vegetableQuantity.html',
                          {'preSchool': preSchool, 'pregnantFAA': pregnantFAA,
                           'lactatingFAA': lactatingFAA,
                           'preSchoolGlv': preSchoolGlv, 'preSchoolRt': preSchoolRt, 'preSchoolOveg': preSchoolOveg,
                           'preSchoolFruit': preSchoolFruit, 'preSchoolCs': preSchoolCs,

                           'pregnantGlv': pregnantGlv, 'pregnantRt': pregnantRt, 'pregnantOveg': pregnantOveg,
                           'pregnantFruit': pregnantFruit,
                           'pregnantCs': pregnantCs,

                           'lactatingGlv': lactatingGlv, 'lactatingRt': lactatingRt,
                           'lactatingOveg': lactatingOveg, 'lactatingFruit': lactatingFruit, 'lactatingCs': lactatingCs
                           })

    def post(self, request):
        preSchool_Bal = request.POST.get('preSchool_Bathua leaves', None)
        preSchool_Fel = request.POST.get('preSchool_Fenugreek leaves', None)
        preSchool_Drl = request.POST.get('preSchool_Drumstick leaves', None)
        preSchool_Col = request.POST.get('preSchool_Colocasia leaves', None)
        preSchool_Aml = request.POST.get('preSchool_Amanranth leaves (red)', None)
        preSchool_Mul = request.POST.get('preSchool_Mustard leaves', None)
        preSchool_Pal = request.POST.get('preSchool_Palak', None)

        preSchool_spo = request.POST.get('preSchool_Sweet potato', None)
        preSchool_yet = request.POST.get('preSchool_Yam elephant', None)
        preSchool_taa = request.POST.get('preSchool_Tapioca', None)
        preSchool_coa = request.POST.get('preSchool_Colocasia', None)
        preSchool_pao = request.POST.get('preSchool_Potato', None)
        preSchool_ono = request.POST.get('preSchool_Onion', None)

        preSchool_pum = request.POST.get('preSchool_Pumpkin (Orange)', None)
        preSchool_bot = request.POST.get('preSchool_Bottle gourd (pale green, elongated)', None)
        preSchool_fre = request.POST.get('preSchool_French beans', None)
        preSchool_cau = request.POST.get('preSchool_Cauliflower', None)
        preSchool_car = request.POST.get('preSchool_Carrot', None)
        preSchool_tom = request.POST.get('preSchool_Tomato', None)
        preSchool_bri = request.POST.get('preSchool_Brinjal (all varieties)', None)
        preSchool_tin = request.POST.get('preSchool_Tinda(apple gourd)', None)

        preSchool_sap = request.POST.get('preSchool_Sapota', None)
        preSchool_ban = request.POST.get('preSchool_Banana', None)
        preSchool_gua = request.POST.get('preSchool_Guava', None)
        preSchool_pap = request.POST.get('preSchool_Papaya', None)
        preSchool_rai = request.POST.get('preSchool_Raisins(dry grapes)', None)

        preSchool_corl = request.POST.get('preSchool_Coriander leaves', None)
        preSchool_gron = request.POST.get('preSchool_Groundnuts', None)
        preSchool_tamp = request.POST.get('preSchool_Tamarind pulp', None)
        preSchool_cums = request.POST.get('preSchool_Cumin seeds', None)
        preSchool_grec = request.POST.get('preSchool_Green chil', None)
        preSchool_ging = request.POST.get('preSchool_Ginger', None)
        preSchool_salt = request.POST.get('preSchool_Salt', None)
        preSchool_curd = request.POST.get('preSchool_Curd', None)
        preSchool_murr = request.POST.get('preSchool_Murmura(puffed rice)', None)
        preSchool_curl = request.POST.get('preSchool_Curry leaves', None)

        pregnant_Bal = request.POST.get('pregnant_Bathua leaves', None)
        pregnant_Fel = request.POST.get('pregnant_Fenugreek leaves', None)
        pregnant_Drl = request.POST.get('pregnant_Drumstick leaves', None)
        pregnant_Col = request.POST.get('pregnant_Colocasia leaves', None)
        pregnant_Aml = request.POST.get('pregnant_Amanranth leaves (red)', None)
        pregnant_Mul = request.POST.get('pregnant_Mustard leaves', None)
        pregnant_Pal = request.POST.get('pregnant_Palak', None)

        pregnant_spo = request.POST.get('pregnant_Sweet potato', None)
        pregnant_yet = request.POST.get('pregnant_Yam elephant', None)
        pregnant_taa = request.POST.get('pregnant_Tapioca', None)
        pregnant_coa = request.POST.get('pregnant_Colocasia', None)
        pregnant_pao = request.POST.get('pregnant_Potato', None)
        pregnant_ono = request.POST.get('pregnant_Onion', None)

        pregnant_pum = request.POST.get('pregnant_Pumpkin (Orange)', None)
        pregnant_bot = request.POST.get('pregnant_Bottle gourd (pale green, elongated)', None)
        pregnant_fre = request.POST.get('pregnant_French beans', None)
        pregnant_cau = request.POST.get('pregnant_Cauliflower', None)
        pregnant_car = request.POST.get('pregnant_Carrot', None)
        pregnant_tom = request.POST.get('pregnant_Tomato', None)
        pregnant_bri = request.POST.get('pregnant_Brinjal (all varieties)', None)
        pregnant_tin = request.POST.get('pregnant_Tinda(apple gourd)', None)

        pregnant_sap = request.POST.get('pregnant_Sapota', None)
        pregnant_ban = request.POST.get('pregnant_Banana', None)
        pregnant_gua = request.POST.get('pregnant_Guava', None)
        pregnant_pap = request.POST.get('pregnant_Papaya', None)
        pregnant_rai = request.POST.get('pregnant_Raisins(dry grapes)', None)

        pregnant_corl = request.POST.get('pregnant_Coriander leaves', None)
        pregnant_gron = request.POST.get('pregnant_Groundnuts', None)
        pregnant_tamp = request.POST.get('pregnant_Tamarind pulp', None)
        pregnant_cums = request.POST.get('pregnant_Cumin seeds', None)
        pregnant_grec = request.POST.get('pregnant_Green chil', None)
        pregnant_ging = request.POST.get('pregnant_Ginger', None)
        pregnant_salt = request.POST.get('pregnant_Salt', None)
        pregnant_curd = request.POST.get('pregnant_Curd', None)
        pregnant_murr = request.POST.get('pregnant_Murmura(puffed rice)', None)
        pregnant_curl = request.POST.get('pregnant_Curry leaves', None)

        lactating_Bal = request.POST.get('lactating_Bathua leaves', None)
        lactating_Fel = request.POST.get('lactating_Fenugreek leaves', None)
        lactating_Drl = request.POST.get('lactating_Drumstick leaves', None)
        lactating_Col = request.POST.get('lactating_Colocasia leaves', None)
        lactating_Aml = request.POST.get('lactating_Amanranth leaves (red)', None)
        lactating_Mul = request.POST.get('lactating_Mustard leaves', None)
        lactating_Pal = request.POST.get('lactating_Palak', None)

        lactating_spo = request.POST.get('lactating_Sweet potato', None)
        lactating_yet = request.POST.get('lactating_Yam elephant', None)
        lactating_taa = request.POST.get('lactating_Tapioca', None)
        lactating_coa = request.POST.get('lactating_Colocasia', None)
        lactating_pao = request.POST.get('lactating_Potato', None)
        lactating_ono = request.POST.get('lactating_Onion', None)

        lactating_pum = request.POST.get('lactating_Pumpkin (Orange)', None)
        lactating_bot = request.POST.get('lactating_Bottle gourd (pale green, elongated)', None)
        lactating_fre = request.POST.get('lactating_French beans', None)
        lactating_cau = request.POST.get('lactating_Cauliflower', None)
        lactating_car = request.POST.get('lactating_Carrot', None)
        lactating_tom = request.POST.get('lactating_Tomato', None)
        lactating_bri = request.POST.get('lactating_Brinjal (all varieties)', None)
        lactating_tin = request.POST.get('lactating_Tinda(apple gourd)', None)

        lactating_sap = request.POST.get('lactating_Sapota', None)
        lactating_ban = request.POST.get('lactating_Banana', None)
        lactating_gua = request.POST.get('lactating_Guava', None)
        lactating_pap = request.POST.get('lactating_Papaya', None)
        lactating_rai = request.POST.get('lactating_Raisins(dry grapes)', None)

        lactating_corl = request.POST.get('lactating_Coriander leaves', None)
        lactating_gron = request.POST.get('lactating_Groundnuts', None)
        lactating_tamp = request.POST.get('lactating_Tamarind pulp', None)
        lactating_cums = request.POST.get('lactating_Cumin seeds', None)
        lactating_grec = request.POST.get('lactating_Green chil', None)
        lactating_ging = request.POST.get('lactating_Ginger', None)
        lactating_salt = request.POST.get('lactating_Salt', None)
        lactating_curd = request.POST.get('lactating_Curd', None)
        lactating_murr = request.POST.get('lactating_Murmura(puffed rice)', None)
        lactating_curl = request.POST.get('lactating_Curry leaves', None)

        preSchool_Qnt_list = dict(
            {'Bathua leaves': preSchool_Bal, 'Fenugreek leaves': preSchool_Fel, 'Drumstick leaves': preSchool_Drl,
             'Colocasia leaves': preSchool_Col,
             'Amanranth leaves (red)': preSchool_Aml, 'Mustard leaves': preSchool_Mul, 'Palak': preSchool_Pal,
             'Sweet potato': preSchool_spo, 'Yam elephant': preSchool_yet, 'Tapioca': preSchool_taa,
             'Colocasia': preSchool_coa, 'Potato': preSchool_pao, 'Onion': preSchool_ono,
             'Pumpkin (Orange)': preSchool_pum, 'Bottle gourd (pale green, elongated)': preSchool_bot,
             'French beans': preSchool_fre, 'Cauliflower': preSchool_cau, 'Carrot': preSchool_car,
             'Tomato': preSchool_tom,
             'Brinjal (all varieties)': preSchool_bri, 'Tinda(apple gourd)': preSchool_tin,
             'Sapota': preSchool_sap, 'Banana': preSchool_ban, 'Guava': preSchool_gua, 'Papaya': preSchool_pap,
             'Raisins(dry grapes)': preSchool_rai,
             'Coriander leaves': preSchool_corl, 'Groundnuts': preSchool_gron, 'Tamarind pulp': preSchool_tamp,
             'Cumin seeds': preSchool_cums, 'Green chil': preSchool_grec,
             'Ginger': preSchool_ging, 'Salt': preSchool_salt, 'Curd': preSchool_curd,
             'Murmura(puffed rice)': preSchool_murr, 'Curry leaves': preSchool_curl
             })

        preSchool_Qnt_list = dict(filter(lambda item: item[1] is not None, preSchool_Qnt_list.items()))

        pregnant_Qnt_list = dict(
            {'Bathua leaves': pregnant_Bal, 'Fenugreek leaves': pregnant_Fel, 'Drumstick leaves': pregnant_Drl,
             'Colocasia leaves': pregnant_Col,
             'Amanranth leaves (red)': pregnant_Aml, 'Mustard leaves': pregnant_Mul, 'Palak': pregnant_Pal,
             'Sweet potato': pregnant_spo, 'Yam elephant': pregnant_yet, 'Tapioca': pregnant_taa,
             'Colocasia': pregnant_coa, 'Potato': pregnant_pao, 'Onion': pregnant_ono,
             'Pumpkin (Orange)': pregnant_pum, 'Bottle gourd (pale green, elongated)': pregnant_bot,
             'French beans': pregnant_fre, 'Cauliflower': pregnant_cau, 'Carrot': pregnant_car, 'Tomato': pregnant_tom,
             'Brinjal (all varieties)': pregnant_bri, 'Tinda(apple gourd)': pregnant_tin,
             'Sapota': pregnant_sap, 'Banana': pregnant_ban, 'Guava': pregnant_gua, 'Papaya': pregnant_pap,
             'Raisins(dry grapes)': pregnant_rai,
             'Coriander leaves': pregnant_corl, 'Groundnuts': pregnant_gron, 'Tamarind pulp': pregnant_tamp,
             'Cumin seeds': pregnant_cums, 'Green chil': pregnant_grec,
             'Ginger': pregnant_ging, 'Salt': pregnant_salt, 'Curd': pregnant_curd,
             'Murmura(puffed rice)': pregnant_murr, 'Curry leaves': pregnant_curl
             })

        pregnant_Qnt_list = dict(filter(lambda item: item[1] is not None, pregnant_Qnt_list.items()))

        lactating_Qnt_list = dict(
            {'Bathua leaves': lactating_Bal, 'Fenugreek leaves': lactating_Fel, 'Drumstick leaves': lactating_Drl,
             'Colocasia leaves': lactating_Col,
             'Amanranth leaves (red)': lactating_Aml, 'Mustard leaves': lactating_Mul, 'Palak': lactating_Pal,
             'Sweet potato': lactating_spo, 'Yam elephant': lactating_yet, 'Tapioca': lactating_taa,
             'Colocasia': lactating_coa, 'Potato': lactating_pao, 'Onion': lactating_ono,
             'Pumpkin (Orange)': lactating_pum, 'Bottle gourd (pale green, elongated)': lactating_bot,
             'French beans': lactating_fre, 'Cauliflower': lactating_cau, 'Carrot': lactating_car,
             'Tomato': lactating_tom,
             'Brinjal (all varieties)': lactating_bri, 'Tinda(apple gourd)': lactating_tin,
             'Sapota': lactating_sap, 'Banana': lactating_ban, 'Guava': lactating_gua, 'Papaya': lactating_pap,
             'Raisins(dry grapes)': lactating_rai,
             'Coriander leaves': lactating_corl, 'Groundnuts': lactating_gron, 'Tamarind pulp': lactating_tamp,
             'Cumin seeds': lactating_cums, 'Green chil': lactating_grec,
             'Ginger': lactating_ging, 'Salt': lactating_salt, 'Curd': lactating_curd,
             'Murmura(puffed rice)': lactating_murr, 'Curry leaves': lactating_curl
             })

        lactating_Qnt_list = dict(filter(lambda item: item[1] is not None, lactating_Qnt_list.items()))

        request.session['preSchool_Qnt_list'] = preSchool_Qnt_list
        request.session['pregnant_Qnt_list'] = pregnant_Qnt_list
        request.session['lactating_Qnt_list'] = lactating_Qnt_list
        return redirect('veg_result')


class VegResult(View):

    def get(self, request):
        if ('query' not in request.session) and ('infant' not in request.session):
            return redirect('home')
        if request.session['query'] == 'THR':
            return redirect('home')
        if request.session['query'] == 'FAA':
            if 'preSchool' not in request.session:
                return redirect('home')

            preSchool = request.session.get('preSchool', None)
            pregnantFAA = request.session.get('pregnantFAA', None)
            lactatingFAA = request.session.get('lactatingFAA', None)
            Cereals = request.session.get('Cereals', None)
            Pulses = request.session.get('Pulses', None)
            Others = request.session.get('Others', None)
            scheme = request.session.get('scheme', None)
            milkPowderQuantity = request.session.get('milkPowderQuantity', None)
            if milkPowderQuantity is not None:
                milkPowderQuantity = int(milkPowderQuantity)
            cereal_prop = []
            pulse_prop = []
            other_prop = []
            final_q = ''
            if request.session['preSchool'] > 0:
                cost_list = request.session['cost_list']
                for k, v in cost_list.items():
                    cost_list[k] = float(v)

                cost_list = pd.DataFrame(cost_list.items(), columns=['Food_Name', 'input_cost'])
                print(cost_list)

                preSchoolFood = request.session['preSchoolFood']
                print(preSchoolFood)
                preSchoolFoodCost = cost_list[cost_list['Food_Name'].isin(preSchoolFood)]
                print(preSchoolFoodCost)

                preSchoolFoodCost['input_cost'] = preSchoolFoodCost['input_cost'] / 1000
                Age_group = "4yrs-6yrs"
                Food = preSchoolFoodCost['Food_Name']

            return render(request, 'icds/resultAddVegetable.html',
                          {'module': 'FAA', 'preSchool': preSchool, 'pregnantFAA': pregnantFAA,
                           'lactatingFAA': lactatingFAA,
                           'Cereals': Cereals, 'Pulses': Pulses, 'Others': Others, 'cereal_prop': cereal_prop,
                           'pulse_prop': pulse_prop, 'other_prop': other_prop
                           })

        return render(request, 'icds/index.html')


def filter_vegetable_data(request):
    category_id = request.GET.get("category_id")
    print(category_id)
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

        if request.session.get('pregnantFAA', None):
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

            query = request.session['query']
            print(query)
            final_out_pregnant = LPPWOVARHCM(Age_group, Food, pregnantFoodCost, pregnantscheme,
                                             pregnantmilkPowderQuantity)
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

            pregnant_Qnt_list = request.session.get('pregnant_Qnt_list', None)
            print(pregnant_Qnt_list)

            pregnant_Qnt_df = pd.DataFrame(pregnant_Qnt_list.items())
            pregnant_Qnt_df.columns = ['Food_Name', 'Amount']

            nutrition_calc = VEGNUTCAL(pregnant_Qnt_df)
            print(nutrition_calc)

            temp_nut = NUTCAL(final_out_pregnant)
            print(temp_nut)
            df_add = pd.DataFrame()
            df_add["Nutritions"] = temp_nut["Nutritions"]
            df_add["Amount"] = temp_nut["Amount"] + nutrition_calc["Amount"]
            print(df_add)

            final_q = df_add.set_index('Nutritions')['Amount'].to_dict()
            print(opStatus)
            if opStatus == 'Infeasible':
                p_Infeasible = render_to_string('icds/infeasible.html')
                return JsonResponse({'data': p_Infeasible})

            p = render_to_string('icds/resultVegPregnant.html',
                                 {
                                     'pregnantCereals': pregnantCereals, 'pregnantPulses': pregnantPulses,
                                     'pregnantOthers': pregnantOthers, 'cereal_prop_pregnant': cereal_prop_pregnant,
                                     'pulse_prop_pregnant': pulse_prop_pregnant,
                                     'other_prop_pregnant': other_prop_pregnant,
                                     'milk_prop_pregnant': milk_prop_pregnant,
                                     'final_q': final_q, 'final_optimized_cost': final_optimized_cost,
                                     'pregnantscheme': pregnantscheme,
                                     'opStatus': opStatus, 'pregnant_Qnt_list': pregnant_Qnt_list
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

        if request.session.get('lactatingFAA', None):

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

            query = request.session['query']
            print(query)
            final_out_lactating = LPPWOVARHCM(Age_group, Food, lactatingFoodCost, lactatingscheme,
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

            lactating_Qnt_list = request.session.get('lactating_Qnt_list', None)
            print(lactating_Qnt_list)

            lactating_Qnt_df = pd.DataFrame(lactating_Qnt_list.items())
            lactating_Qnt_df.columns = ['Food_Name', 'Amount']

            nutrition_calc = VEGNUTCAL(lactating_Qnt_df)
            print(nutrition_calc)

            temp_nut = NUTCAL(final_out_lactating)
            print(temp_nut)
            df_add = pd.DataFrame()
            df_add["Nutritions"] = temp_nut["Nutritions"]
            df_add["Amount"] = temp_nut["Amount"] + nutrition_calc["Amount"]
            print(df_add)

            final_q = df_add.set_index('Nutritions')['Amount'].to_dict()

            print(final_q, opStatus)
            if opStatus == 'Infeasible':
                lw_Infeasible = render_to_string('icds/infeasible.html')
                return JsonResponse({'data': lw_Infeasible})

            lw = render_to_string('icds/resultVegLactating.html',
                                  {
                                      'lactatingCereals': lactatingCereals, 'lactatingPulses': lactatingPulses,
                                      'lactatingOthers': lactatingOthers,
                                      'cereal_prop_lactating': cereal_prop_lactating,
                                      'pulse_prop_lactating': pulse_prop_lactating,
                                      'other_prop_lactating': other_prop_lactating,
                                      'milk_prop_lactating': milk_prop_lactating,
                                      'final_q': final_q, 'final_optimized_cost': final_optimized_cost,
                                      'lactatingscheme': lactatingscheme,
                                      'opStatus': opStatus, 'lactating_Qnt_list': lactating_Qnt_list
                                  })
            return JsonResponse({'data': lw})

    if category_id == '4 yrs to 6 yrs':
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

        if request.session['preSchool'] > 0:
            cost_list = request.session['cost_list']
            for k, v in cost_list.items():
                cost_list[k] = float(v)

            cost_list = pd.DataFrame(cost_list.items(), columns=['Food_Name', 'input_cost'])
            print(cost_list)

            preSchoolFood = request.session['preSchoolFood']
            print(preSchoolFood)
            infantFoodCost = cost_list[cost_list['Food_Name'].isin(preSchoolFood)]
            print(infantFoodCost)

            infantFoodCost['input_cost'] = infantFoodCost['input_cost'] / 1000
            Age_group = "child(4-6)yrs"
            Food = infantFoodCost['Food_Name']

            final_out_infant = LPPWOVARHCM(Age_group, Food, infantFoodCost, scheme, milkPowderQuantity)

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

            preSchool_Qnt_list = request.session.get('preSchool_Qnt_list', None)
            print(preSchool_Qnt_list)
            preSchool_Qnt_df = pd.DataFrame(preSchool_Qnt_list.items())
            preSchool_Qnt_df.columns = ['Food_Name', 'Amount']

            nutrition_calc = VEGNUTCAL(preSchool_Qnt_df)
            print(nutrition_calc)
            temp_nut = NUTCAL(final_out_infant)
            print(temp_nut)

            df_add = pd.DataFrame()
            df_add["Nutritions"] = temp_nut["Nutritions"]
            df_add["Amount"] = temp_nut["Amount"] + nutrition_calc["Amount"]
            print(df_add)

            final_q = df_add.set_index('Nutritions')['Amount'].to_dict()
            print(final_q)
            if opStatus == 'Infeasible':
                i_Infeasible = render_to_string('icds/infeasible.html')
                return JsonResponse({'data': i_Infeasible})
            i = render_to_string('icds/resultVegPreSchool.html',
                                 {
                                     'Cereals': Cereals, 'Pulses': Pulses, 'Others': Others, 'milkpowder': milkpowder,
                                     'cereal_prop': cereal_prop, 'pulse_prop': pulse_prop, 'other_prop': other_prop,
                                     'milk_prop': milk_prop,
                                     'final_q': final_q, 'final_optimized_cost': final_optimized_cost,
                                     'opStatus': opStatus, 'scheme': scheme, 'preSchool_Qnt_list': preSchool_Qnt_list
                                 })
            return JsonResponse({'data': i})
