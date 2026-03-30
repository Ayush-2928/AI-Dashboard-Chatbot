from django.shortcuts import render

def index(request):
    return render(request, 'dashboard/index.html')

def dashboard2(request):
    return render(request, 'dashboard/dashboard2.html')

def sales_cockpit3(request):
    return render(request, 'dashboard/sales_cockpit3.html')

def wireframe(request):
    return render(request, 'dashboard/wireframe.html')

def sales_cockpit(request):
    return render(request, 'dashboard/sales_cockpit.html')
