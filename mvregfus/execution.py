# import pickle
# client = Client(processes=True)
import sys, os
from distributed import Client

client = Client(processes=False)  # ,threads_per_worker=1)
# client = Client(processes=False, threads_per_worker=10)
# client = Client(processes=False,threads_per_worker=1)
dashboard_link = 'http://localhost:%s' % int(client.cluster.scheduler.service_ports['dashboard'])
print('LINK TO DASHBOARD: dashboard_link')

if sys.platform.startswith("win"):
    try:
        os.system("title " + "multi-view fusion: " + dashboard_link)
    except:
        pass
elif sys.platform.startswith("lin") or sys.platform.startswith("dar"):
    print('\33]0;multi-view fusion: %s\a' % dashboard_link, end='', flush=True)