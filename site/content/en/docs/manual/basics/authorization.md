---
title: 'Authorization'
linkTitle: 'Authorization'
weight: 1
---

- First of all, you have to log in to CVAT tool. For authentication, you can use your username or email
  you provided during registration.

  ![](/images/image001.jpg)

- For register a new user press "Create an account"

  ![](/images/image002.jpg)

- You can register a user but by default it will not have rights even to view
  list of tasks. Thus you should create a superuser. The superuser can use
  [Django administration panel](http://localhost:8080/admin) to assign correct
  groups to the user. Please use the command below to create an admin account:

  ```bash
    docker exec -it cvat_server bash -ic 'python3 ~/manage.py createsuperuser'
  ```

- If you want to create a non-admin account, you can do that using the link below
  on the login page. Don't forget to modify permissions for the new user in the
  administration panel. There are several groups (aka roles): admin, user,
  annotator, observer.

  ![](/images/image003.jpg)

  A username generates from the email automatically. It can be edited if needed.

  ![](/images/filling_email.gif)

### Administration panel

Go to the [Django administration panel](http://localhost:8080/admin). There you can:

- Create / edit / delete users
- Control permissions of users and access to the tool.

  ![](/images/image115.jpg)
